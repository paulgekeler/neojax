"""Tests for dataset downloading utilities."""

import os
import tempfile

import pytest
import requests

from neojax.data.download.dataset import download_dataset
from neojax.data.download.dataverse_downloader import DataverseDownloader
from neojax.data.download.http_downloader import HTTPDownloader
from neojax.data.download.huggingface_downloader import HuggingFaceDownloader
from neojax.data.download.zenodo_downloader import ZenodoDownloader


class TestHTTPDownloader:
    def test_http_downloader_success(self, mocker):
        """Test successful download and checksum validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response = mocker.MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-length": "10"}
            mock_response.iter_content.return_value = [b"data1", b"data2"]

            mock_get = mocker.patch("requests.get", return_value=mock_response)
            urls = {"test_file.txt": "http://example.com/test_file.txt"}
            # md5 of b"data1data2" is "ec7df88ac3eb6c69121cf62eb8231217"
            checksums = {"test_file.txt": "md5:ec7df88ac3eb6c69121cf62eb8231217"}
            downloader = HTTPDownloader(urls=urls, checksums=checksums)

            filepaths = downloader.download(tmpdir)

            assert len(filepaths) == 1
            assert os.path.basename(filepaths[0]) == "test_file.txt"
            with open(filepaths[0], "rb") as f:
                assert f.read() == b"data1data2"

            mock_get.assert_called_once_with(
                "http://example.com/test_file.txt",
                stream=True,
                headers={"User-Agent": "neojax-downloader/0.1"},
                timeout=30,
            )

    def test_http_downloader_checksum_mismatch(self, mocker):
        """Test download failure when checksum doesn't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_response = mocker.MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-length": "10"}
            mock_response.iter_content.return_value = [b"data1", b"data2"]

            urls = {"test_file.txt": "http://example.com/test_file.txt"}
            # Incorrect md5
            checksums = {"test_file.txt": "md5:incorrecthash"}
            downloader = HTTPDownloader(urls=urls, checksums=checksums, max_retries=1)

            with pytest.raises((requests.RequestException, ValueError, IOError)):
                downloader.download(tmpdir)

    def test_http_downloader_resume(self, mocker):
        """Test resume functionality using Range header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_file.txt")
            part_filepath = filepath + ".part"
            # Write initial part bytes
            with open(part_filepath, "wb") as f:
                f.write(b"data1")

            mock_response = mocker.MagicMock()
            mock_response.status_code = 206
            mock_response.headers = {"content-length": "5"}
            mock_response.iter_content.return_value = [b"data2"]

            mock_get = mocker.patch("requests.get", return_value=mock_response)
            urls = {"test_file.txt": "http://example.com/test_file.txt"}
            checksums = {"test_file.txt": "md5:ec7df88ac3eb6c69121cf62eb8231217"}
            downloader = HTTPDownloader(urls=urls, checksums=checksums)

            filepaths = downloader.download(tmpdir)

            assert len(filepaths) == 1
            with open(filepaths[0], "rb") as f:
                assert f.read() == b"data1data2"

            mock_get.assert_called_once_with(
                "http://example.com/test_file.txt",
                stream=True,
                headers={
                    "User-Agent": "neojax-downloader/0.1",
                    "Range": "bytes=5-",
                },
                timeout=30,
            )


class TestZenodoDownloader:
    def test_zenodo_downloader_success(self, mocker):
        """Test Zenodo record parsing and HTTP downloader delegation."""
        mock_get = mocker.patch("requests.get")
        mock_record_response = mocker.MagicMock()
        mock_record_response.status_code = 200
        mock_record_response.json.return_value = {
            "files": [
                {
                    "key": "file1.h5",
                    "checksum": "md5:12345",
                    "links": {"content": "http://zenodo.org/file1.h5"},
                }
            ]
        }
        mock_get.return_value = mock_record_response

        downloader = ZenodoDownloader(record_id="12345", filenames=["file1.h5"])

        MockHTTPDownloader = mocker.patch(
            "neojax.data.download.zenodo_downloader.HTTPDownloader"
        )
        mock_http_instance = MockHTTPDownloader.return_value
        mock_http_instance.download.return_value = ["/tmp/file1.h5"]

        paths = downloader.download("/tmp/target")

        assert paths == ["/tmp/file1.h5"]
        MockHTTPDownloader.assert_called_once_with(
            urls={"file1.h5": "http://zenodo.org/file1.h5"},
            checksums={"file1.h5": "md5:12345"},
            max_retries=5,
            backoff_factor=1.5,
        )


class TestHuggingFaceDownloader:
    def test_huggingface_downloader_success(self, mocker):
        """Test HF api file discovery, filtering, and URL resolution."""
        mock_get = mocker.patch("requests.get")
        mock_hf_response = mocker.MagicMock()
        mock_hf_response.status_code = 200
        mock_hf_response.json.return_value = {
            "siblings": [
                {"rpath": ".gitattributes"},
                {"rpath": "README.md"},
                {"rpath": "data.h5"},
                {"rpath": "nested/data2.h5"},
            ]
        }
        mock_get.return_value = mock_hf_response

        downloader = HuggingFaceDownloader(repo_id="camlab-ethz/NS-Gauss")

        MockHTTPDownloader = mocker.patch(
            "neojax.data.download.huggingface_downloader.HTTPDownloader"
        )
        mock_http_instance = MockHTTPDownloader.return_value
        mock_http_instance.download.return_value = [
            "/tmp/target/data.h5",
            "/tmp/target/nested/data2.h5",
        ]

        paths = downloader.download("/tmp/target")

        assert paths == [
            "/tmp/target/data.h5",
            "/tmp/target/nested/data2.h5",
        ]
        MockHTTPDownloader.assert_called_once_with(
            urls={
                "data.h5": (
                    "https://huggingface.co/datasets/camlab-ethz/"
                    "NS-Gauss/resolve/main/data.h5"
                ),
                "nested/data2.h5": (
                    "https://huggingface.co/datasets/camlab-ethz/"
                    "NS-Gauss/resolve/main/nested/data2.h5"
                ),
            },
            max_retries=5,
            backoff_factor=1.5,
        )


class TestDownloadDataset:
    def test_download_dataset_from_registry(self, mocker):
        """Test download_dataset resolving dataset names from registry."""
        MockHFDownloader = mocker.patch(
            "neojax.data.download.dataset.HuggingFaceDownloader"
        )
        mock_hf_instance = MockHFDownloader.return_value
        mock_hf_instance.download.return_value = ["/tmp/ace_data.h5"]

        paths = download_dataset("ace", "/tmp/target")

        assert paths == ["/tmp/ace_data.h5"]
        MockHFDownloader.assert_called_once_with(repo_id="camlab-ethz/ACE")

    def test_download_dataset_dataverse(self, mocker):
        """Test download_dataset resolving dataverse downloader."""
        MockDataverseDownloader = mocker.patch(
            "neojax.data.download.dataset.DataverseDownloader"
        )
        mock_instance = MockDataverseDownloader.return_value
        mock_instance.download.return_value = ["/tmp/pdebench_data.hdf5"]

        paths = download_dataset("pdebench_sod_1d", "/tmp/target")

        assert paths == ["/tmp/pdebench_data.hdf5"]
        MockDataverseDownloader.assert_called_once_with(
            doi="10.18419/darus-2986", filenames=["Sod6.hdf5"]
        )

    def test_download_dataset_invalid_name(self):
        """Test error raised when requesting unregistered dataset name."""
        with pytest.raises(ValueError) as excinfo:
            download_dataset("nonexistent_dataset_abc", "/tmp/target")
        assert "is not registered" in str(excinfo.value)


class TestDataverseDownloader:
    def test_dataverse_downloader_success(self, mocker):
        """Test successful Dataverse API file discovery and download delegation."""
        mock_get = mocker.patch("requests.get")
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "files": [
                    {
                        "label": "file1.hdf5",
                        "dataFile": {"id": 12345, "md5": "abc123md5hash"},
                    },
                    {
                        "label": "file2.hdf5",
                        "dataFile": {
                            "id": 67890,
                            "checksum": {"type": "MD5", "value": "def456md5hash"},
                        },
                    },
                ]
            }
        }
        mock_get.return_value = mock_response

        downloader = DataverseDownloader(
            doi="10.18419/darus-2986", filenames=["file1.hdf5", "file2.hdf5"]
        )

        MockHTTPDownloader = mocker.patch(
            "neojax.data.download.dataverse_downloader.HTTPDownloader"
        )
        mock_http_instance = MockHTTPDownloader.return_value
        mock_http_instance.download.return_value = [
            "/tmp/target/file1.hdf5",
            "/tmp/target/file2.hdf5",
        ]

        paths = downloader.download("/tmp/target")

        assert paths == ["/tmp/target/file1.hdf5", "/tmp/target/file2.hdf5"]
        MockHTTPDownloader.assert_called_once_with(
            urls={
                "file1.hdf5": "https://darus.uni-stuttgart.de/api/access/datafile/12345",
                "file2.hdf5": "https://darus.uni-stuttgart.de/api/access/datafile/67890",
            },
            checksums={
                "file1.hdf5": "md5:abc123md5hash",
                "file2.hdf5": "md5:def456md5hash",
            },
            max_retries=5,
            backoff_factor=1.5,
        )
