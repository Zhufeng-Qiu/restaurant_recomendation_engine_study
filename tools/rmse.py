"""RMSE of cf_predict.py output against test_review_ratings.json.

Follows the INF553 HW3 grading protocol: test pairs the model could not
predict fall back to the business average (item-based) or user average
(user-based), with the "UNK" entry for cold-start IDs.

Usage:
    python3 tools/rmse.py <predictions.jsonl> <test_ratings.json> <avg.json> <item|user>
"""

import json
import math
import sys


def main():
    pred_path, truth_path, avg_path, mode = sys.argv[1:5]
    with open(avg_path) as f:
        avg = json.load(f)

    preds = {}
    with open(pred_path) as f:
        for line in f:
            j = json.loads(line)
            preds[(j["user_id"], j["business_id"])] = j["stars"]

    n = 0
    covered = 0
    sq = 0.0
    with open(truth_path) as f:
        for line in f:
            j = json.loads(line)
            key = (j["user_id"], j["business_id"])
            truth = float(j["stars"])
            if key in preds:
                p = preds[key]
                covered += 1
            else:
                fallback_id = j["business_id"] if mode == "item" else j["user_id"]
                p = avg.get(fallback_id, avg["UNK"])
            sq += (p - truth) ** 2
            n += 1
    print(f"pairs={n} predicted={covered} coverage={covered/n:.4f} "
          f"RMSE={math.sqrt(sq/n):.6f}")


if __name__ == "__main__":
    main()
