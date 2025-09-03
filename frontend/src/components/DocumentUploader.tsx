import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Button,
  Box,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
} from '@mui/material';
import { CloudUpload, CheckCircle, Error, Article } from '@mui/icons-material';
import { ragService } from '../services/ragService';
import { Corpus } from '../types';

interface DocumentUploaderProps {
  corpora: Corpus[];
  selectedCorpus: string;
  onCorpusSelect: (corpusId: string) => void;
  onUploadComplete: () => void;
}

interface FileUploadStatus {
  name: string;
  status: 'pending' | 'uploading' | 'success' | 'error';
  message?: string;
}

export default function DocumentUploader({ 
  corpora, 
  selectedCorpus, 
  onCorpusSelect,
  onUploadComplete 
}: DocumentUploaderProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<FileUploadStatus[]>([]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const selectedFiles = Array.from(event.target.files);
      setFiles(selectedFiles);
      setUploadStatus(selectedFiles.map(file => ({
        name: file.name,
        status: 'pending'
      })));
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedCorpus) {
      setError('Please select a corpus');
      return;
    }

    if (files.length === 0) {
      setError('Please select files to upload');
      return;
    }

    setUploading(true);
    setError(null);

    const newStatus = [...uploadStatus];

    for (let i = 0; i < files.length; i++) {
      newStatus[i] = { ...newStatus[i], status: 'uploading' };
      setUploadStatus([...newStatus]);

      try {
        await ragService.uploadDocument(selectedCorpus, files[i]);
        newStatus[i] = { ...newStatus[i], status: 'success' };
      } catch (err: any) {
        newStatus[i] = { 
          ...newStatus[i], 
          status: 'error',
          message: err.message || 'Upload failed'
        };
      }
      setUploadStatus([...newStatus]);
    }

    setUploading(false);
    onUploadComplete();
  };

  const handleClear = () => {
    setFiles([]);
    setUploadStatus([]);
    setError(null);
  };

  const getStatusIcon = (status: FileUploadStatus['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle color="success" />;
      case 'error':
        return <Error color="error" />;
      case 'uploading':
        return <CircularProgress size={20} />;
      default:
        return <Article />;
    }
  };

  const getFileExtension = (filename: string) => {
    return filename.split('.').pop()?.toUpperCase() || 'FILE';
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Document Upload
      </Typography>

      <FormControl fullWidth sx={{ mb: 3 }}>
        <InputLabel>Select Corpus</InputLabel>
        <Select
          value={selectedCorpus}
          label="Select Corpus"
          onChange={(e) => onCorpusSelect(e.target.value)}
          disabled={uploading}
        >
          <MenuItem value="">
            <em>None</em>
          </MenuItem>
          {corpora.map((corpus) => (
            <MenuItem key={corpus.id} value={corpus.id}>
              {corpus.name} ({corpus.document_count} documents)
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Box sx={{ mb: 3 }}>
        <input
          accept=".txt,.pdf,.json,.md,.csv,.xml,.html"
          style={{ display: 'none' }}
          id="file-upload"
          multiple
          type="file"
          onChange={handleFileSelect}
          disabled={uploading}
        />
        <label htmlFor="file-upload">
          <Button
            variant="outlined"
            component="span"
            startIcon={<CloudUpload />}
            disabled={uploading}
            fullWidth
          >
            Select Files
          </Button>
        </label>
      </Box>

      {files.length > 0 && (
        <>
          <Typography variant="subtitle1" gutterBottom>
            Selected Files ({files.length})
          </Typography>
          <List sx={{ mb: 2 }}>
            {uploadStatus.map((file, index) => (
              <ListItem key={index}>
                <ListItemIcon>
                  {getStatusIcon(file.status)}
                </ListItemIcon>
                <ListItemText 
                  primary={file.name}
                  secondary={file.message}
                />
                <Chip 
                  label={getFileExtension(file.name)}
                  size="small"
                  variant="outlined"
                />
              </ListItem>
            ))}
          </List>

          {uploading && (
            <LinearProgress sx={{ mb: 2 }} />
          )}

          <Box display="flex" gap={2}>
            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={uploading || !selectedCorpus}
              fullWidth
            >
              {uploading ? 'Uploading...' : 'Upload Documents'}
            </Button>
            <Button
              variant="outlined"
              onClick={handleClear}
              disabled={uploading}
            >
              Clear
            </Button>
          </Box>
        </>
      )}
    </Paper>
  );
}

function CircularProgress({ size }: { size: number }) {
  return (
    <Box sx={{ display: 'flex' }}>
      <Box sx={{ 
        width: size, 
        height: size,
        border: '2px solid',
        borderColor: 'primary.main',
        borderRadius: '50%',
        borderTopColor: 'transparent',
        animation: 'spin 1s linear infinite',
        '@keyframes spin': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' }
        }
      }} />
    </Box>
  );
}