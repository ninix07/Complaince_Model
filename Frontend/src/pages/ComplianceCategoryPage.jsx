/** @format */

import axios from "axios";
import { useState } from "react";

const ComplianceCategoryPage = () => {
  const [text, setText] = useState("");
  const [response, setResponse] = useState("");
  const [id, setId] = useState("");
  const [err, setErr] = useState("")
  const handleComplianceClick = async () => {
    const requestBody = {
      q_id: id,
      question: text,
    };
    console.log(text);
    try {
      const res = await axios.post(
        "http://127.0.0.1:8080/api/getComplaince",
        requestBody,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      console.log(res);
      setResponse(`Compliance: ${res.data.Compliance}`);
    } catch (error) {
      setErr(error.response.data.error)
      console.error("Error fetching compliance data", error.response.data.error);
    }
  };

  const handleCategoryClick = async () => {
    const requestBody = {
      q_id: id,
      question: text,
    };
    console.log(text);
    try {
      const res = await axios.post(
        "http://127.0.0.1:8080/api/predictCategory",
        requestBody,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      setResponse(
        `Category: ${res.data.Category}\nSub-Category :${res.data.Sub_Category}`
      );
      // setResponse(`Compliance: ${res.data.ComplCiance}`); // Update response state
    } catch (error) {
      setErr(error.response.data.error)
      console.error("Error fetching compliance data", error.response.data.error);
    }
  };

  return (
    <div className="flex justify-center items-center h-[80vh]">
      <div className="bg-white p-8 rounded-lg shadow-xl w-full max-w-md">
        <h1 className="text-2xl font-semibold text-center mb-6 text-gray-800">
          Compliance and Category Selector
        </h1>

        <div className="relative mb-4">
          <input
            type="text"
            value={id}
            onChange={(e) => { setId(e.target.value); setResponse(""); }}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-blue-400 bg-black focus:border-blue-400"
            placeholder="Question Id"
          />
          <input
            type="text"
            value={text}
            onChange={(e) => { setText(e.target.value); setResponse(""); }}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-blue-400 bg-black focus:border-blue-400"
            placeholder="Please Insert you Question"
          />
        </div>

        <div className="flex space-x-4">
          <button
            onClick={handleComplianceClick}
            className="w-1/2 bg-blue-600 text-white py-2 rounded-lg shadow-md hover:bg-blue-700 transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none"
          >
            Compliance
          </button>
          <button
            onClick={handleCategoryClick}
            className="w-1/2 bg-green-600 text-white py-2 rounded-lg shadow-md hover:bg-green-700 transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none"
          >
            Category
          </button>
        </div>

        {response && (
          <div className="mt-4 text-center text-lg font-semibold text-gray-700">
            <pre>{response}</pre>
          </div>
        )}
        {err && (
          <div className="mt-4 text-center text-lg font-semibold text-red-700">
            <pre className="whitespace-pre-wrap break-words">Error:{err}</pre>
          </div>
        )

        }
      </div>
    </div>
  );
};

export default ComplianceCategoryPage;
