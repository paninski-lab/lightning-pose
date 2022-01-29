def test_dfConverter():
    import pandas as pd

    df = pd.read_csv(
        "toy_datasets/toymouseRunningData/CollectedData_.csv", header=[1, 2]
    )
    from lightning_pose.utils.fiftyone import dfConverter

    out = dfConverter(df)()
    assert type(out) is dict
    # print(df.head())
