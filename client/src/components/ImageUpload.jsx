import React, { useState, useRef } from "react";
import { 
  Box, 
  Typography, 
  Button, 
  Card, 
  CardMedia, 
  CardContent, 
  Container, 
  CircularProgress,
  TextField,
  Fade,
  Chip,
  Dialog
} from "@mui/material";
import { styled } from "@mui/material/styles";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import SearchIcon from "@mui/icons-material/Search";
import ImageIcon from "@mui/icons-material/Image";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";

const MagicalContainer = styled(Container)(() => ({
  background: 'linear-gradient(135deg, #faf7ff 0%, #f8faff 50%, #fefaff 100%)',
  minHeight: '100vh',
  padding: '2rem 0',
}));

const MagicalCard = styled(Card)(() => ({
  background: 'rgba(255,255,255,0.85)',
  backdropFilter: 'blur(10px)',
  borderRadius: '16px',
  border: '1px solid rgba(147, 51, 234, 0.08)',
  boxShadow: '0 4px 20px rgba(147, 51, 234, 0.08)',
  overflow: 'visible',
}));

const MagicalButton = styled(Button)(() => ({
  background: 'linear-gradient(45deg, #a855f7 0%, #8b5cf6 100%)',
  border: 0,
  borderRadius: '12px',
  color: 'white',
  height: 48,
  padding: '0 24px',
  boxShadow: '0 3px 12px rgba(168, 85, 247, 0.2)',
  fontSize: '1rem',
  fontWeight: 500,
  textTransform: 'none',
  transition: 'all 0.2s ease',
  '&:hover': {
    background: 'linear-gradient(45deg, #9333ea 0%, #7c3aed 100%)',
    boxShadow: '0 4px 16px rgba(168, 85, 247, 0.25)',
    transform: 'translateY(-1px)',
  },
  '&:disabled': {
    background: '#e2e8f0',
    boxShadow: 'none',
  }
}));

const UploadButton = styled(Button)(() => ({
  background: 'rgba(168, 85, 247, 0.04)',
  border: '2px dashed rgba(168, 85, 247, 0.3)',
  borderRadius: '12px',
  color: '#7c3aed',
  height: 100,
  fontSize: '1rem',
  fontWeight: 500,
  textTransform: 'none',
  transition: 'all 0.2s ease',
  '&:hover': {
    background: 'rgba(168, 85, 247, 0.08)',
    borderColor: 'rgba(168, 85, 247, 0.4)',
  }
}));

const MagicalTextField = styled(TextField)(() => ({
  '& .MuiOutlinedInput-root': {
    borderRadius: '12px',
    background: 'rgba(255, 255, 255, 0.7)',
    '& fieldset': {
      borderColor: 'rgba(168, 85, 247, 0.2)',
    },
    '&:hover fieldset': {
      borderColor: 'rgba(168, 85, 247, 0.3)',
    },
    '&.Mui-focused fieldset': {
      borderColor: '#a855f7',
    },
  },
  '& .MuiInputLabel-root': {
    color: '#6b7280',
    '&.Mui-focused': {
      color: '#7c3aed',
    }
  },
}));

const MainResultSection = styled(Box)(() => ({
  background: 'rgba(255,255,255,0.6)',
  borderRadius: '16px',
  padding: '24px',
  border: '1px solid rgba(168, 85, 247, 0.1)',
  height: 'fit-content',
}));

const SidebarSection = styled(Box)(() => ({
  background: 'rgba(255,255,255,0.4)',
  borderRadius: '12px',
  padding: '20px',
  border: '1px solid rgba(168, 85, 247, 0.06)',
  height: 'fit-content',
  position: 'sticky',
  top: '20px',
}));

const StepSection = styled(Box)(() => ({
  background: 'rgba(255,255,255,0.5)',
  borderRadius: '12px',
  padding: '16px',
  border: '1px solid rgba(168, 85, 247, 0.06)',
  marginBottom: '16px',
  transition: 'all 0.2s ease',
  '&:hover': {
    transform: 'translateX(4px)',
    boxShadow: '0 4px 12px rgba(168, 85, 247, 0.1)',
  }
}));

const SectionTitle = styled(Typography)(() => ({
  color: '#374151',
  fontWeight: 600,
  textAlign: 'center',
  marginBottom: '16px',
}));

const StepChip = styled(Chip)(() => ({
  background: 'rgba(168, 85, 247, 0.08)',
  color: '#6b7280',
  fontWeight: 500,
  borderRadius: '8px',
  border: '1px solid rgba(168, 85, 247, 0.1)',
  marginBottom: '12px',
  '& .MuiChip-icon': {
    color: '#a855f7',
  }
}));

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [word, setWord] = useState("");
  const [previewUrl, setPreviewUrl] = useState(null);
  const [processedImageUrl, setProcessedImageUrl] = useState(null);
  const [bwImageUrl, setBwImageUrl] = useState(null);
  const [smudgedImageUrl, setSmudgedImageUrl] = useState(null);
  const [extractedImageUrl, setExtractedImageUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedStepImage, setSelectedStepImage] = useState(null);
  const [openStepDialog, setOpenStepDialog] = useState(false);
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
      setBwImageUrl(`data:image/png;base64,${result.bw_image_base64}`);
      setSmudgedImageUrl(`data:image/png;base64,${result.smudged_image_base64}`);
      setExtractedImageUrl(`data:image/png;base64,${result.extracted_image_base64}`);
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleImageClick = (image) => {
    setSelectedStepImage(image);
    setOpenStepDialog(true);
  };

  const handleCloseDialog = () => {
    setSelectedStepImage(null);
    setOpenStepDialog(false);
  };

  const hasResults = bwImageUrl || smudgedImageUrl || extractedImageUrl || processedImageUrl;

  return (
    <MagicalContainer maxWidth="xl">
      <MagicalCard>
        <CardContent sx={{ p: 4 }}>
          <Box textAlign="center" mb={4}>
            <Typography variant="h4" sx={{ color: '#a855f7', fontWeight: 600, mb: 1 }}>
              Word Detection System
            </Typography>
            <Typography variant="body1" color="#6b7280">
              Upload an image and search for specific words
            </Typography>
          </Box>

          <Box display="flex" flexDirection="column" gap={3} mb={4}>
            <MagicalTextField
              fullWidth
              label="Enter word to search"
              value={word}
              onChange={handleWordChange}
              variant="outlined"
              InputProps={{
                startAdornment: <SearchIcon sx={{ color: '#9ca3af', mr: 1 }} />
              }}
            />

            <Box onClick={() => fileInputRef.current.click()}>
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileChange}
                style={{ display: "none" }}
                accept="image/*"
              />
              <UploadButton fullWidth>
                <CloudUploadIcon sx={{ mr: 2 }} />
                <Box>
                  <Typography variant="body1" component="div">
                    Choose an image file
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.7 }}>
                    Click to browse or drag and drop
                  </Typography>
                </Box>
              </UploadButton>
            </Box>
          </Box>

          <Box display="flex" justifyContent="center" mb={4}>
            <MagicalButton
              onClick={handleUpload}
              disabled={!file || !word || isUploading}
              size="large"
              startIcon={isUploading ? 
                <CircularProgress size={18} sx={{ color: "white" }} /> : 
                <SearchIcon />
              }
            >
              {isUploading ? "Processing..." : "Find Word"}
            </MagicalButton>
          </Box>

          {previewUrl && !hasResults && (
            <Fade in={true} timeout={600}>
              <MainResultSection>
                <SectionTitle variant="h6">Original Image</SectionTitle>
                <CardMedia 
                  component="img" 
                  image={previewUrl} 
                  alt="Preview" 
                  sx={{ borderRadius: '8px', width: '100%', height: 'auto' }} 
                />
              </MainResultSection>
            </Fade>
          )}

          {hasResults && (
            <Box display="flex" gap={4}>
              <Box flex="0 0 70%">
                {previewUrl && (
                  <Fade in={true} timeout={600}>
                    <MainResultSection sx={{ mb: 4 }}>
                      <SectionTitle variant="h6">Original Image</SectionTitle>
                      <CardMedia 
                        component="img" 
                        image={previewUrl} 
                        alt="Preview" 
                        sx={{ borderRadius: '8px', width: '100%', height: 'auto' }} 
                      />
                    </MainResultSection>
                  </Fade>
                )}

                {processedImageUrl && (
                  <Fade in={true} timeout={1400}>
                    <MainResultSection>
                      <SectionTitle variant="h5" >Detection Result</SectionTitle>
                      <CardMedia 
                        component="img" 
                        image={processedImageUrl} 
                        alt="Final Result" 
                        sx={{ borderRadius: '8px', width: '100%', height: 'auto', border: '2px solid rgba(168, 85, 247, 0.2)' }} 
                      />
                    </MainResultSection>
                  </Fade>
                )}
              </Box>

              <Box flex="0 0 30%">
                <SidebarSection>
                  <Typography variant="h6" sx={{ color: '#374151', fontWeight: 600, mb: 3, textAlign: 'center' }}>
                    Processing Steps
                  </Typography>

                  {bwImageUrl && (
                    <Fade in={true} timeout={800}>
                      <StepSection>
                        <Box textAlign="center" mb={2}>
                          <StepChip icon={<ImageIcon />} label="B&W Conversion" size="small" />
                        </Box>
                        <Box onClick={() => handleImageClick(bwImageUrl)} sx={{ cursor: 'pointer' }}>
                          <CardMedia 
                            component="img" 
                            image={bwImageUrl} 
                            alt="B/W" 
                            sx={{ borderRadius: '8px', width: '100%', height: 'auto', border: '1px solid rgba(168, 85, 247, 0.1)' }} 
                          />
                        </Box>
                      </StepSection>
                    </Fade>
                  )}

                  {smudgedImageUrl && (
                    <Fade in={true} timeout={1000}>
                      <StepSection>
                        <Box textAlign="center" mb={2}>
                          <StepChip icon={<AutoFixHighIcon />} label="Smudging" size="small" />
                        </Box>
                        <Box onClick={() => handleImageClick(smudgedImageUrl)} sx={{ cursor: 'pointer' }}>
                          <CardMedia 
                            component="img" 
                            image={smudgedImageUrl} 
                            alt="Enhanced" 
                            sx={{ borderRadius: '8px', width: '100%', height: 'auto', border: '1px solid rgba(168, 85, 247, 0.1)' }} 
                          />
                        </Box>
                      </StepSection>
                    </Fade>
                  )}

                  {extractedImageUrl && (
                    <Fade in={true} timeout={1200}>
                      <StepSection>
                        <Box textAlign="center" mb={2}>
                          <StepChip icon={<SearchIcon />} label="Word Extraction" size="small" />
                        </Box>
                        <Box onClick={() => handleImageClick(extractedImageUrl)} sx={{ cursor: 'pointer' }}>
                          <CardMedia 
                            component="img" 
                            image={extractedImageUrl} 
                            alt="Extracted" 
                            sx={{ borderRadius: '8px', width: '100%', height: 'auto', border: '1px solid rgba(168, 85, 247, 0.1)' }} 
                          />
                        </Box>
                      </StepSection>
                    </Fade>
                  )}
                </SidebarSection>
              </Box>
            </Box>
          )}

          <Dialog
            open={openStepDialog}
            onClose={handleCloseDialog}
            maxWidth="md"
            fullWidth
            PaperProps={{
              sx: {
                borderRadius: '16px',
                p: 2,
                background: '#fff',
                boxShadow: '0 4px 24px rgba(147, 51, 234, 0.2)',
              }
            }}
          >
            <Box display="flex" flexDirection="column" alignItems="center">
              <Typography variant="h6" mb={2} sx={{ fontWeight: 600, color: '#6b21a8' }}>
                Processing Step Preview
              </Typography>
              {selectedStepImage && (
                <Box component="img" src={selectedStepImage} alt="Step Detail" sx={{
                  maxWidth: '100%',
                  borderRadius: '12px',
                  border: '1px solid rgba(147, 51, 234, 0.1)'
                }} />
              )}
              <Button 
                onClick={handleCloseDialog} 
                variant="outlined" 
                sx={{ mt: 3, textTransform: 'none', color: '#9333ea', borderColor: '#9333ea' }}
              >
                Close
              </Button>
            </Box>
          </Dialog>
        </CardContent>
      </MagicalCard>
    </MagicalContainer>
  );
};

export default ImageUpload;
