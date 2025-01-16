import io
import zipfile
from pathlib import Path

import requests


def fetch_test_data_if_needed(dir: Path, dataset_name: str) -> None:
    datasets_url_dict = {
        "test_cropzoom_data": "https://figshare.com/ndownloader/files/51015435"
    }
    # check if data exists
    dataset_dir = dir / dataset_name
    # TODO Add a way to force download fresh data.
    # Maybe compare file size of stored dataset and figshare dataset?
    # Figshare filesize can be gotten with HEAD request, and stored
    # in the dataset directory.
    if dataset_dir.exists():
        return

    url = datasets_url_dict[dataset_name]
    print(f"Fetching {dataset_name} from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for download errors
        with zipfile.ZipFile(io.BytesIO(r.raw.read())) as z:
            # Extract assuming there is only one directory in the zip file.
            file_list = z.namelist()
            top_level_file_list = [name for name in file_list if name.count("/") <= 1]
            if (
                len(top_level_file_list) > 1
                or top_level_file_list[0] != f"{dataset_name}/"
            ):
                raise ValueError(
                    f"Zip file must have only one dir called {dataset_dir}\n"
                    f"Instead found {file_list}."
                )
            else:
                z.extractall(dir)

    print("Done")
