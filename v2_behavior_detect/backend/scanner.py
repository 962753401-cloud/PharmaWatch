"""YOLO-World + Supervision video scanner with three behavior triggers."""
from __future__ import annotations

import logging
import numpy as np
import supervision as sv
import cv2
import ultralytics
from pathlib import Path
from typing import Callable, Optional

from config import SETTINGS

logger = logging.getLogger("v2_scanner")


class TriggerEvent:
    """A detected behavioral trigger event."""

    def __init__(self, timestamp: float, trigger_type: str, details: str):
        self.timestamp = timestamp
        self.trigger_type = trigger_type
        self.details = details

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 1),
            "trigger_type": self.trigger_type,
            "details": self.details,
        }


class FrameAnnotation:
    """An annotated frame for live preview."""

    def __init__(self, frame_idx: int, timestamp: float, jpeg_bytes: bytes,
                 person_count: int, object_count: int):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.jpeg_bytes = jpeg_bytes
        self.person_count = person_count
        self.object_count = object_count


class VideoScanner:
    """Scans a video for behavioral triggers using YOLO-World + Supervision."""

    def __init__(
        self,
        video_path: str | Path,
        on_progress: Optional[Callable] = None,
        on_trigger: Optional[Callable] = None,
        on_frame: Optional[Callable] = None,
    ):
        self.video_path = str(video_path)
        self.on_progress = on_progress
        self.on_trigger = on_trigger
        self.on_frame = on_frame
        self._stop = False

    def stop(self):
        self._stop = True

    def scan(self):
        """Run the full scan. Calls on_progress, on_trigger, and on_frame callbacks."""
        cfg = SETTINGS

        # Configure ultralytics to use our local weights directory (absolute path)
        weights_dir = cfg.weights_path.parent
        try:
            ultralytics.settings.update({"weights_dir": str(weights_dir)})
        except Exception:
            pass  # settings.json may be read-only; the in-memory update still works

        # Patch the text_model module's WEIGHTS_DIR so CLIP loads from local cache
        import ultralytics.nn.text_model as _tm
        _tm.WEIGHTS_DIR = weights_dir
        logger.info("Set weights_dir=%s, patched text_model.WEIGHTS_DIR", weights_dir)

        from ultralytics import YOLOWorld
        logger.info("Loading YOLO-World model from %s", cfg.weights_path)
        model = YOLOWorld(str(cfg.weights_path))
        model.set_classes(list(cfg.classes))
        logger.info("Model loaded. Classes: %s", list(cfg.classes))

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        logger.info("Video: fps=%.2f frames=%d duration=%.1fs", fps, total_frames, duration)

        sample_interval = cfg.sample_interval
        sampled_fps = fps / sample_interval
        logger.info("Sampling 1 frame per %d (effective fps=%.1f)", sample_interval, sampled_fps)

        # Build class name -> class_id mapping
        class_name_to_id = {name: i for i, name in enumerate(cfg.classes)}
        id_to_name = {v: k for k, v in class_name_to_id.items()}
        person_cls_id = class_name_to_id.get("person", 0)
        object_cls_ids = set(
            class_name_to_id[n] for n in cfg.object_classes if n in class_name_to_id
        )
        chair_cls_ids = set(
            class_name_to_id[n] for n in cfg.chair_classes if n in class_name_to_id
        )
        carry_target_ids = set(
            class_name_to_id[n] for n in cfg.carry_target_classes if n in class_name_to_id
        )

        # ByteTrack for person tracking (deprecated but functional in 0.29.0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=cfg.lost_track_buffer,
                minimum_matching_threshold=0.8,
                frame_rate=sampled_fps,
                minimum_consecutive_frames=1,
            )

        # Supervision annotators for live preview
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Trigger state
        crowd_consecutive = 0
        carry_consecutive = 0
        # tracker_id -> list of (cx, cy, is_sitting, timestamp)
        tracker_history: dict[int, list[tuple[float, float, bool, float]]] = {}
        last_trigger_time: dict[str, float] = {"crowd": -999, "carry": -999, "loiter": -999}

        frame_idx = 0
        sampled_count = 0

        while True:
            if self._stop:
                logger.info("Scan stopped by user at frame %d", frame_idx)
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                current_time = frame_idx / fps
                sampled_count += 1

                # Run YOLO-World detection
                results = model.predict(frame, conf=cfg.conf_object, verbose=False)
                r = results[0]
                detections = sv.Detections.from_ultralytics(r)

                # --- Split detections by category ---
                person_dets = sv.Detections.empty()
                object_dets = sv.Detections.empty()
                chair_dets = sv.Detections.empty()
                carry_dets = sv.Detections.empty()

                if len(detections) > 0:
                    person_mask = (detections.class_id == person_cls_id) & (
                        detections.confidence >= cfg.conf_person
                    )
                    person_dets = detections[person_mask]

                    obj_mask_list = []
                    for cid in object_cls_ids:
                        obj_mask_list.append(detections.class_id == cid)
                    if obj_mask_list:
                        obj_combined = obj_mask_list[0]
                        for m in obj_mask_list[1:]:
                            obj_combined = obj_combined | m
                        obj_combined = obj_combined & (detections.confidence >= cfg.conf_object)
                        object_dets = detections[obj_combined]

                    # Carry trigger only looks at target classes (phone)
                    carry_mask_list = []
                    for cid in carry_target_ids:
                        carry_mask_list.append(detections.class_id == cid)
                    if carry_mask_list:
                        carry_combined = carry_mask_list[0]
                        for m in carry_mask_list[1:]:
                            carry_combined = carry_combined | m
                        carry_combined = carry_combined & (detections.confidence >= cfg.conf_object)
                        carry_dets = detections[carry_combined]

                    chair_mask_list = []
                    for cid in chair_cls_ids:
                        chair_mask_list.append(detections.class_id == cid)
                    if chair_mask_list:
                        chair_combined = chair_mask_list[0]
                        for m in chair_mask_list[1:]:
                            chair_combined = chair_combined | m
                        chair_combined = chair_combined & (detections.confidence >= cfg.conf_object)
                        chair_dets = detections[chair_combined]

                person_count = len(person_dets)

                # --- Trigger 1: Crowd (person count >= threshold for consecutive frames) ---
                if person_count >= cfg.crowd_threshold:
                    crowd_consecutive += 1
                    if crowd_consecutive >= cfg.crowd_consecutive:
                        if current_time - last_trigger_time["crowd"] >= cfg.cooldown:
                            last_trigger_time["crowd"] = current_time
                            evt = TriggerEvent(
                                current_time, "crowd",
                                f"\u68c0\u6d4b\u5230{person_count}\u4eba\u805a\u96c6"
                            )
                            logger.info("CROWD trigger at %.1fs (%d persons)", current_time, person_count)
                            if self.on_trigger:
                                self.on_trigger(evt)
                else:
                    crowd_consecutive = 0

                # --- Trigger 2: Person + phone sustained for carry_sustain_frames ---
                carry_found = False
                carry_objs = []
                if len(person_dets) > 0 and len(carry_dets) > 0:
                    carry_found, carry_objs = self._check_carry(person_dets, carry_dets, cfg, id_to_name)

                if carry_found:
                    carry_consecutive += 1
                    if carry_consecutive >= cfg.carry_sustain_frames:
                        if current_time - last_trigger_time["carry"] >= cfg.cooldown:
                            last_trigger_time["carry"] = current_time
                            unique_objs = list(dict.fromkeys(carry_objs))
                            evt = TriggerEvent(
                                current_time, "carry",
                                f"\u4eba+{chr(43).join(unique_objs)}\u6301\u7eed{carry_consecutive}\u5e27"
                            )
                            logger.info("CARRY trigger at %.1fs (phone near person, %d frames)", current_time, carry_consecutive)
                            if self.on_trigger:
                                self.on_trigger(evt)
                else:
                    carry_consecutive = 0

                # --- Trigger 3: Sitting + duration (person on chair for >= loiter_seconds) ---
                if len(person_dets) > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        tracked = tracker.update_with_detections(person_dets)

                    sitting_ids = set()
                    if len(chair_dets) > 0:
                        for i in range(len(tracked)):
                            tid = int(tracked.tracker_id[i])
                            px1, py1, px2, py2 = tracked.xyxy[i]
                            p_cx = float((px1 + px2) / 2)
                            p_cy = float((py1 + py2) / 2)

                            is_sitting = False
                            for j in range(len(chair_dets)):
                                cx1, cy1, cx2, cy2 = chair_dets.xyxy[j]
                                iou = self._box_iou(
                                    [float(px1), float(py1), float(px2), float(py2)],
                                    [float(cx1), float(cy1), float(cx2), float(cy2)],
                                )
                                if iou >= 0.1:
                                    is_sitting = True
                                    break
                                c_cx = (cx1 + cx2) / 2
                                c_cy = (cy1 + cy2) / 2
                                dist = float(np.sqrt((p_cx - c_cx) ** 2 + (p_cy - c_cy) ** 2))
                                if dist <= cfg.loiter_radius:
                                    is_sitting = True
                                    break

                            if is_sitting:
                                sitting_ids.add(tid)

                            if tid not in tracker_history:
                                tracker_history[tid] = []
                            tracker_history[tid].append((p_cx, p_cy, is_sitting, current_time))

                            cutoff = current_time - 30
                            tracker_history[tid] = [
                                (x, y, s, t) for x, y, s, t in tracker_history[tid] if t >= cutoff
                            ]

                            hist = tracker_history[tid]
                            if len(hist) >= 2 and is_sitting:
                                sitting_duration = 0.0
                                for k in range(len(hist) - 1, 0, -1):
                                    if hist[k][2] and hist[k - 1][2]:
                                        sitting_duration = hist[k][3] - hist[k - 1][3] + sitting_duration
                                    elif hist[k][2]:
                                        sitting_duration = 0.0
                                        continue
                                    else:
                                        break

                                if sitting_duration >= cfg.loiter_seconds:
                                    if current_time - last_trigger_time["loiter"] >= cfg.cooldown:
                                        last_trigger_time["loiter"] = current_time
                                        evt = TriggerEvent(
                                            current_time, "loiter",
                                            f"ID{tid}\u5750\u59ff\u505c\u7559{sitting_duration:.1f}\u79d2"
                                        )
                                        logger.info(
                                            "LOITER trigger at %.1fs (ID%d, %.1fs sitting)",
                                            current_time, tid, sitting_duration,
                                        )
                                        if self.on_trigger:
                                            self.on_trigger(evt)
                                        tracker_history[tid] = [(p_cx, p_cy, is_sitting, current_time)]

                    stale = [tid for tid, h in tracker_history.items() if not h or h[-1][3] < current_time - 60]
                    for tid in stale:
                        del tracker_history[tid]

                # --- Annotate frame for live preview ---
                if self.on_frame:
                    annotated = self._annotate_frame(
                        frame, detections, id_to_name, cfg,
                        current_time, person_count, len(object_dets),
                        box_annotator, label_annotator,
                    )
                    self.on_frame(FrameAnnotation(
                        frame_idx=sampled_count,
                        timestamp=current_time,
                        jpeg_bytes=annotated,
                        person_count=person_count,
                        object_count=len(object_dets),
                    ))

                # Progress callback
                if self.on_progress:
                    self.on_progress(frame_idx, total_frames, current_time, duration)

            frame_idx += 1

        cap.release()
        logger.info("Scan complete. Processed %d frames (%d sampled).", frame_idx, sampled_count)

    def _annotate_frame(self, frame, detections, id_to_name, cfg,
                        current_time, person_count, object_count,
                        box_annotator, label_annotator):
        """Annotate frame with detection boxes and labels, return JPEG bytes."""
        annotated = frame.copy()

        if len(detections) > 0:
            # Build labels with class name + confidence
            labels = []
            for i in range(len(detections)):
                cid = int(detections.class_id[i])
                conf = float(detections.confidence[i])
                name = id_to_name.get(cid, str(cid))
                labels.append(f"{name} {conf:.0%}")

            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # Overlay timestamp and counts in top-left corner
        h, w = annotated.shape[:2]
        ts_text = f"t={current_time:.1f}s  persons={person_count}  objects={object_count}"
        cv2.rectangle(annotated, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(annotated, ts_text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Resize if too wide
        max_w = cfg.annotated_max_width
        if w > max_w:
            scale = max_w / w
            annotated = cv2.resize(annotated, (max_w, int(h * scale)))

        # Encode as JPEG
        ok, buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, cfg.annotated_jpeg_quality])
        if not ok:
            return b""
        return buf.tobytes()

    def _check_carry(self, person_dets, carry_dets, cfg, id_to_name):
        """Check if any carry-target detection center falls within an expanded person box.
        Returns (bool, list of object class names found)."""
        expand = cfg.carry_expand
        found_objs = []
        for i in range(len(person_dets)):
            px1, py1, px2, py2 = person_dets.xyxy[i]
            px1 -= expand
            py1 -= expand
            px2 += expand
            py2 += expand
            for j in range(len(carry_dets)):
                ox1, oy1, ox2, oy2 = carry_dets.xyxy[j]
                ocx = (ox1 + ox2) / 2
                ocy = (oy1 + oy2) / 2
                if px1 <= ocx <= px2 and py1 <= ocy <= py2:
                    cid = int(carry_dets.class_id[j])
                    name = id_to_name.get(cid, str(cid))
                    found_objs.append(name)
        return (len(found_objs) > 0, found_objs)

    @staticmethod
    def _box_iou(box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        if union <= 0:
            return 0.0
        return intersection / union
