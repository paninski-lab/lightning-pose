
def test_dfConverter():

    import pandas as pd
    from lightning_pose.utils.fiftyone import dfConverter

    df = pd.read_csv(
        "toy_datasets/toymouseRunningData/CollectedData_.csv", header=[1, 2]
    )
    keypoint_names = [c[0] for c in df.columns[1::2]]

    out = dfConverter(df, keypoint_names=keypoint_names)()
    assert type(out) is dict
