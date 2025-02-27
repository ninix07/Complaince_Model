/** @format */

import { useEffect, useState } from "react";
import LoadingAnimation from "../components/Loading.jsx";

const HomePage = () => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploaded, setUploaded] = useState(false); // Track upload success

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      console.log("File selected:", selectedFile);
      setFile(selectedFile);
      setUploaded(false); // Reset upload state when a new file is selected
    }
  };

  useEffect(() => {
    console.log("UseEffect File selected:", file);
  }, [file]);

  const handleUpload = async () => {
    console.log("Uploading file:", file);
    setError(null);
    setUploaded(false); // Reset success state
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8080/api/createTaxonomy", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(
          `Upload failed: ${response.status} ${response.statusText}`
        );
      }

      const responseData = await response.json();
      setUploaded(true);
      console.log("File uploaded successfully:", responseData);
    } catch (error) {
      console.error("Error uploading file:", error);
      setError(error.message || "An unknown error occurred while uploading.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center w-full relative">
      {/* Full-page loading overlay */}
      {loading && <LoadingAnimation />}

      <label
        htmlFor="dropzone-file"
        className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-gray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600"
      >
        <div className="flex flex-col items-center justify-center pt-5 pb-6">
          <svg
            className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400"
            aria-hidden="true"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 20 16"
          >
            <path
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
            />
          </svg>
          <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
            <span className="font-semibold">Click to upload</span> or drag and
            drop
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            CSV File
          </p>
        </div>
        <input
          id="dropzone-file"
          type="file"
          className="hidden"
          onChange={handleFileChange}
        />
      </label>

      {file && (
        <div className="mt-4 flex flex-col items-center">
          <p className="text-sm text-gray-600">Selected file: {file.name}</p>
          <button
            onClick={handleUpload}
            disabled={loading}
            className={`mt-2 px-4 py-2 ${loading ? "bg-blue-300" : "bg-blue-500 hover:bg-blue-600"
              } text-white rounded-lg flex items-center justify-center transition-colors`}
          >
            Upload CSV
          </button>
        </div>
      )}


      {error && (
        <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      {uploaded && (
        <div className="mt-4 p-3 bg-green-100 text-green-700 rounded-lg flex items-center">
          <svg
            className="w-5 h-5 mr-2"
            fill="currentColor"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            ></path>
          </svg>
          File uploaded successfully!
        </div>
      )}
    </div>
  );
};

export default HomePage;
