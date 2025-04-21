import React, { useState, useRef } from "react";
import { Box, Typography, Button, Card, CardMedia, CardContent, Container, CircularProgress } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [word, setWord] = useState("");
  const [previewUrl, setPreviewUrl] = useState(null);
  const [processedImageUrl, setProcessedImageUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const fileReader = new FileReader();
      fileReader.onload = () => setPreviewUrl(fileReader.result);
      fileReader.readAsDataURL(selectedFile);
    }
  };

  const handleWordChange = (e) => setWord(e.target.value);

  const handleUpload = async () => {
    if (!file || !word) return;
    setIsUploading(true);

    const formData = new FormData();
    formData.append("image", file);
    formData.append("word", word);

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      setProcessedImageUrl(`data:image/png;base64,${result.image_base64}`);

    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ my: 5 }}>
      <Card>
        <CardContent>
          <Typography variant="h4" align="center" sx={{ fontWeight: 700 }}>
            Image Upload & Template Matching
          </Typography>

          <input
            type="text"
            value={word}
            onChange={handleWordChange}
            placeholder="Enter word to match"
            style={{ margin: "10px 0", padding: "8px", width: "100%" }}
          />

          <Box onClick={() => fileInputRef.current.click()}>
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              style={{ display: "none" }}
              accept="image/*"
            />
            <Button variant="contained" color="primary" sx={{ width: "100%" }}>
              <CloudUploadIcon sx={{ mr: 1 }} />
              Upload Image
            </Button>
          </Box>

          {previewUrl && <CardMedia component="img" image={previewUrl} alt="Preview" sx={{ mt: 3 }} />}

          <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
            <Button
              onClick={handleUpload}
              disabled={!file || !word || isUploading}
              variant="contained"
              color="success"
            >
              {isUploading ? <CircularProgress size={20} sx={{ color: "white", mr: 1 }} /> : "Find Word"}
            </Button>
          </Box>

          {processedImageUrl && (
            <Box sx={{ mt: 4, textAlign: "center" }}>
              <Typography variant="h5">Result</Typography>
              <CardMedia component="img" image={processedImageUrl} alt="Processed" sx={{ mt: 2 }} />
            </Box>
          )}
        </CardContent>
      </Card>
    </Container>
  );
};

export default ImageUpload;
