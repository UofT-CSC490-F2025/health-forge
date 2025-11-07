from baseline_llm_modal import app, run_baseline_llm

if __name__ == "__main__":
    with app.run():
        handle = run_baseline_llm.remote(
            s3_bucket="health-forge-data-processing",
            dataset_key="ehr_norm.npy",
            config_key="train_cfg.yaml",
            workdir_key="workdirs/judge_train/samples/all_x.npy",  # optional, remove if not uploaded
        )
        for log in handle.stream_logs():
            print(log, end="")
        handle.wait()
