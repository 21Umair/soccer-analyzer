import numpy as np
from dataclasses import dataclass, field

def cosine_similarity(a, b):
    """Here I am comparing the two feature vectors.
    
    Result:
    - close to 1.0 means very similar
    - close to 0.0 means not similar
    """

    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)

    return float(np.dot(a, b))

@dataclass
class PlayerMemory:
    "This sotres the information about one player identity."

    global_id: int
    last_seen_frame: int
    last_bbox: list
    features: list = field(default_factory=list)
    active_tracker_id: int | None=None


class GlobalIDManager:
    """This class manages our stable player IDs.
    
    tracker_id comes from ByteTrack/BotSort.
    global_id comes from this class"""

    def __int__(
        self,
        primary_ttl_frames= 250,
        secondary_ttl_frames= 3000,
        match_threshold= 0.72,
        max_features_per_player= 20,
    ):
        self.primary_ttl_frames = primary_ttl_frames
        self.secondary_ttl_frames = secondary_ttl_frames
        self.match_threshold = match_threshold
        self.max_features_per_player = max_features_per_player

        self.memories = {}
        self.tracker_to_global = {}
        self.next_global_id = 1

    def _best_feature_score(self, query_feature, memory):
        """
        Compare the current player feature with all the saved features of an old player.
        
        we keep the best score."""

        if len(memory.features) == 0:
            return 0.0

        scores = [
            cosine_similarity(query_feature, stored_feature)
            for stored_feature in memory.features
        ]
        return max(scores)
    
    def _time_score(self, current_frame, last_seen_frame):
        """A player seen recently is more likely to be the same player."""

        gap = current_frame - last_seen_frame

        if gap <= self.primary_ttl_frames:
            return 1.0
        
        if gap> self.secondary_ttl_frames:
            return 0.0
        
        return 1.0 - ((gap - self.primary_ttl_frames) / self.secondary_ttl_frames)
    
    def _bbox_distance_score(self, current_bbox, last_bbox):
        """Check if the new player appears near the old player's last position
        
        This is the simple pixel-distance logic.
        Later can be imroved by using pitch coordinates."""

        cx1 = (current_bbox[0] + current_bbox[2]) / 2
        cy1 = (current_bbox[1] + current_bbox[3]) / 2

        cx2 = (last_bbox[0] + last_bbox[2]) / 2
        cy2 = (last_bbox[1] + last_bbox[3]) / 2

        distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

        #have to be tuned according to the video resolution
        max_resonable_distance = 600

        score = 1.0 - min(distance / max_resonable_distance, 1.0)

        return float(score)
    

    def find_match(self, tracker_id, frame_idx, bbox, feature):
        """
        Main function 
        - decides is this tracker_id already known?
        - decides is this a reappearing old player?
        - or is this a new player."""

        # Case 1:
        # this tracker_id is already connected to a global_id.
        # So just return the same global_id
        if tracker_id in self.tracker_to_global:
            return self.tracker_to_global[tracker_id]
        
        best_global_id = None
        best_score = 0.0

        # Case 2
        # New tracker_id, but maybe it belongs to an old player.
        for global_id, memory in self.memories.items():

            # Do not assign the same global_id to two currently visible players.
            if memory.active_tracker_id is not None:
                continue

            frame_gap = frame_idx - memory.last_seen_frame

            # Too old, ignore it.
            if frame_gap> self.secondary_ttl_frames:
                continue

            appearance_score = self._best_feature_score(feature, memory)
            time_score = self._time_score(frame_idx, memory.last_seen_frame)
            position_score = self._bbox_distance_score(bbox, memory.last_bbox)

            final_score = (0.65 * appearance_score + 0.20 * position_score + 0.15 * time_score)

            if final_score > best_score:
                best_score = final_score
                best_global_id = global_id

        
        # if score is good enough, reuse old global_id.
        if best_score >= self.match_threshold:
            self.tracker_to_global[tracker_id] = best_global_id
            self.memories[best_global_id].active_tracker_id = tracker_id
            return best_global_id
        

        # Case 3
        # No good match found, so crate a new global_id.
        global_id = self.next_global_id
        self.next_global_id += 1
        self.memories[global_id] = PlayerMemory(
            global_id=global_id,
            last_seen_frame=frame_idx,
            last_bbox=bbox,
            features=[feature],
            active_tracker_id=tracker_id,
        )

        self.tracker_to_global[tracker_id] = global_id
        return global_id
    
def update(self, tracker_id, global_id, frame_idx, bbox, feature):
    "Update the memory of this player"

    memory = self.memories[global_id]

    memory.last_seen_frame = frame_idx
    memory.last_bbox = bbox
    memory.active_tracker_id = tracker_id

    memory.features.append(feature)

    # keep only recent useful features.
    if len(memory.features) > self.max_features_per_player:
        memory.features = memory.features[-self.max_features_per_player:]


def mark_missing_tracker(self, active_tracker_ids):
    """
    At each frame, we check which tracker IDs are visible.
    
    if a tracker_id is no longer visible, we remove the active connection.
    but we keep the player's memory."""

    active_tracker_ids = set(active_tracker_ids)

    for tracker_id, global_id in list (self.tracker_to_global.items()):
        if tracker_id not in active_tracker_ids:
            if global_id in self.memories:
                self.memories[global_id].active_tracker_id = None

            del self.tracker_to_global[tracker_id]
