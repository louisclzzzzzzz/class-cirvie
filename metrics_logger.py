import csv
import os
from datetime import datetime

class PerformanceLogger:
    def __init__(self, filename="performance_runs.csv"):
        self.filename = filename
        self.headers = [
            "timestamp", 
            "run_id", 
            "pipeline", 
            "model_type", 
            "omv_acc", "omv_f1",
            "svc_acc", "svc_f1",
            "ori_acc", "ori_f1",
            "comment"
        ]
        self._initialize_file()

    def _initialize_file(self):
        # If file exists but has old headers, we might want to recreate it or handle it.
        # Given the request for a radical change in format, we'll overwrite or start fresh.
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_run_wide(self, run_id, pipeline, model_type, results_dict, comment=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def get_m(target, metric):
            res = results_dict.get(target, {})
            # For SERVICE, use hybrid metrics if available
            if target == "SERVICE" and "metrics_hybrid" in res:
                m = res["metrics_hybrid"]
            else:
                m = res.get("metrics", {})
            
            val = m.get(metric, 0)
            return f"{val:.4f}" if val else "N/A"

        row = [
            timestamp,
            run_id,
            pipeline,
            model_type,
            get_m("OMV", "accuracy"), get_m("OMV", "f1_macro"),
            get_m("SERVICE", "accuracy"), get_m("SERVICE", "f1_macro"),
            get_m("ORIGINE", "accuracy"), get_m("ORIGINE", "f1_macro"),
            comment
        ]
        
        with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def log_results(results, pipeline_name="main.py", model_type="TF-IDF + LogReg", comment=""):
    logger = PerformanceLogger()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # If results is a single target result, wrap it
    if "metrics" in results or "accuracy" in results:
        target = results.get("label", "Unknown")
        results = {target: results}

    logger.log_run_wide(
        run_id=run_id,
        pipeline=pipeline_name,
        model_type=model_type,
        results_dict=results,
        comment=comment
    )
    print(f"  Métriques enregistrées dans {logger.filename}")
