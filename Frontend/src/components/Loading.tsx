/** @format */

import React from "react";

const LoadingAnimation = () => {
  return (
    <div className="fixed inset-0 flex items-center justify-center z-50">
      {/* Semi-transparent backdrop */}
      <div className="absolute inset-0 bg-black bg-opacity-50"></div>

      {/* Loading animation */}
      <div className="flex space-x-2 justify-center items-center z-10">
        <span className="sr-only">Loading...</span>
        <div className="h-8 w-8 bg-white rounded-full animate-bounce [animation-delay:-0.3s]"></div>
        <div className="h-8 w-8 bg-white rounded-full animate-bounce [animation-delay:-0.15s]"></div>
        <div className="h-8 w-8 bg-white rounded-full animate-bounce"></div>
      </div>
    </div>
  );
};

export default LoadingAnimation;
