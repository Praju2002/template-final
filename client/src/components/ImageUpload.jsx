import React, { useState } from "react";
import { uploadImage } from "../api";

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file!");

    const formData = new FormData();
    formData.append("image", file);

    const result = await uploadImage(formData);
    setResponse(result);
  };

  return (
    <div className="flex flex-col items-center gap-4 p-6">
      <h2 className="text-2xl font-bold">Upload Image for Processing</h2>
      <input type="file" onChange={handleFileChange} className="border p-2" />
      <button
        onClick={handleUpload}
        className="bg-blue-500 text-white px-4 py-2 rounded"
      >
        Upload
      </button>

      {response && (
        <div className="mt-4 text-center">
          <h3 className="text-xl">Processed Image:</h3>
          <img src={response.processed_image_url} alt="Processed" className="mt-2 border rounded-lg" />
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
