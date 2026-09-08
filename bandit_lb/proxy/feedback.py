"""Non-blocking asynchronous feedback update pipeline for bandit load balancing."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from bandit_lb.algorithms.base import BaseBanditRouter
from bandit_lb.telemetry.collector import RequestTelemetry, TelemetryCollector
from bandit_lb.telemetry.reward import RewardNormalizer

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """Feedback payload dispatched post-response from the proxy."""

    arm_id: str
    context: np.ndarray[Any, np.dtype[np.float64]]
    latency_ms: float
    is_error: bool = False
    error_type: str | None = None
    ttfb_ms: float | None = None
    status_code: int = 200


class FeedbackPipeline:
    """Asynchronous pipeline decoupling HTTP response returns from bandit matrix math.

    Receives FeedbackEvents via an in-memory queue, computes normalized rewards,
    and updates bandit weights (A_a <- A_a + x x^T) in a dedicated background worker.
    """

    def __init__(
        self,
        router: BaseBanditRouter,
        normalizer: RewardNormalizer | None = None,
        collector: TelemetryCollector | None = None,
        max_queue_size: int = 10000,
        drop_on_full: bool = True,
    ) -> None:
        """Initialize the feedback pipeline.

        Args:
            router: Contextual bandit router instance whose weights will be updated.
            normalizer: RewardNormalizer converting latency into bounded reward.
            collector: Optional TelemetryCollector tracking metrics.
            max_queue_size: Maximum queue capacity.
            drop_on_full: If True, drops event when queue is full rather than blocking.
        """
        self.router = router
        self.normalizer = normalizer or RewardNormalizer()
        self.collector = collector
        self.max_queue_size = max_queue_size
        self.drop_on_full = drop_on_full

        self._queue: asyncio.Queue[FeedbackEvent | None] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: asyncio.Task[None] | None = None
        self._is_running: bool = False

        self.enqueued_count: int = 0
        self.processed_count: int = 0
        self.dropped_count: int = 0

    @property
    def is_running(self) -> bool:
        """Return True if background worker task is active."""
        return self._is_running and self._worker_task is not None and not self._worker_task.done()

    @property
    def queue_size(self) -> int:
        """Return current number of items waiting in queue."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Start background feedback worker task."""
        if self._is_running:
            return
        self._is_running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("FeedbackPipeline worker started (queue capacity: %d)", self.max_queue_size)

    async def stop(self) -> None:
        """Gracefully stop background worker and wait for remaining events to drain."""
        if not self._is_running:
            return
        self._is_running = False
        await self._queue.put(None)
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None
        logger.info(
            "FeedbackPipeline stopped (enqueued: %d, processed: %d, dropped: %d)",
            self.enqueued_count,
            self.processed_count,
            self.dropped_count,
        )

    def dispatch(self, event: FeedbackEvent) -> bool:
        """Dispatch a feedback event asynchronously without blocking the request thread.

        Args:
            event: The FeedbackEvent containing arm_id, context, and latency metrics.

        Returns:
            True if successfully enqueued, False if dropped due to full queue.
        """
        if not self._is_running:
            logger.warning("Attempted to dispatch feedback to inactive pipeline")
            return False

        try:
            self._queue.put_nowait(event)
            self.enqueued_count += 1
            return True
        except asyncio.QueueFull:
            self.dropped_count += 1
            logger.warning(
                "FeedbackPipeline queue full (%d events); dropping event for arm '%s'",
                self.max_queue_size,
                event.arm_id,
            )
            return False

    async def drain(self) -> None:
        """Wait until all currently queued feedback items have been processed."""
        await self._queue.join()

    async def _worker_loop(self) -> None:
        """Dedicated background coroutine consuming events and executing matrix updates."""
        while self._is_running:
            try:
                event = await self._queue.get()
                if event is None:
                    self._queue.task_done()
                    break

                # 1. Compute normalized reward
                reward = self.normalizer.compute_reward(
                    latency_ms=event.latency_ms,
                    is_error=event.is_error,
                )

                # 2. Update contextual bandit online weights
                try:
                    self.router.update(
                        arm_id=event.arm_id,
                        context=event.context,
                        reward=reward,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to update bandit weights for arm '%s': %s", event.arm_id, exc
                    )

                # 3. Record telemetry if collector is attached
                if self.collector is not None:
                    try:
                        self.collector.record(
                            RequestTelemetry(
                                backend_id=event.arm_id,
                                latency_ms=event.latency_ms,
                                ttfb_ms=event.ttfb_ms,
                                status_code=event.status_code,
                                is_error=event.is_error,
                                error_type=event.error_type,
                                context_vector=event.context,
                            )
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to record telemetry for arm '%s': %s", event.arm_id, exc
                        )

                self.processed_count += 1
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Unexpected error in FeedbackPipeline worker: %s", exc)
