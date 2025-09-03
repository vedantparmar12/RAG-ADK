import React, { useState } from 'react';
import {
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Divider,
  Chip,
} from '@mui/material';
import { Search, Clear } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { ragService } from '../services/ragService';
import { Corpus } from '../types';

interface QueryInterfaceProps {
  corpora: Corpus[];
  selectedCorpus: string;
  onCorpusSelect: (corpusId: string) => void;
}

interface QueryResult {
  answer: string;
  sources: Array<{
    text: string;
    metadata: any;
    score: number;
  }>;
  confidence: number;
}

export default function QueryInterface({ corpora, selectedCorpus, onCorpusSelect }: QueryInterfaceProps) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [numResults, setNumResults] = useState(5);

  const handleQuery = async () => {
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }

    if (!selectedCorpus) {
      setError('Please select a corpus');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await ragService.query(selectedCorpus, query, numResults);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Failed to execute query');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuery('');
    setResult(null);
    setError(null);
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Query RAG System
      </Typography>

      <Box sx={{ mb: 3 }}>
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Select Corpus</InputLabel>
          <Select
            value={selectedCorpus}
            label="Select Corpus"
            onChange={(e) => onCorpusSelect(e.target.value)}
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

        <TextField
          fullWidth
          multiline
          rows={3}
          variant="outlined"
          label="Enter your query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          sx={{ mb: 2 }}
        />

        <Box display="flex" gap={2} alignItems="center">
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Results</InputLabel>
            <Select
              value={numResults}
              label="Results"
              onChange={(e) => setNumResults(Number(e.target.value))}
              size="small"
            >
              <MenuItem value={3}>3</MenuItem>
              <MenuItem value={5}>5</MenuItem>
              <MenuItem value={10}>10</MenuItem>
            </Select>
          </FormControl>

          <Button
            variant="contained"
            startIcon={loading ? <CircularProgress size={20} /> : <Search />}
            onClick={handleQuery}
            disabled={loading || !query.trim() || !selectedCorpus}
          >
            {loading ? 'Searching...' : 'Search'}
          </Button>

          <Button
            variant="outlined"
            startIcon={<Clear />}
            onClick={handleClear}
            disabled={loading}
          >
            Clear
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {result && (
        <Box>
          <Divider sx={{ my: 3 }} />
          
          <Card sx={{ mb: 3, bgcolor: 'background.paper' }}>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6" component="h3">
                  Answer
                </Typography>
                {result.confidence && (
                  <Chip 
                    label={`Confidence: ${(result.confidence * 100).toFixed(1)}%`}
                    color={result.confidence > 0.7 ? 'success' : result.confidence > 0.4 ? 'warning' : 'default'}
                  />
                )}
              </Box>
              <Box sx={{ '& p': { mb: 1 } }}>
                <ReactMarkdown>{result.answer}</ReactMarkdown>
              </Box>
            </CardContent>
          </Card>

          {result.sources && result.sources.length > 0 && (
            <>
              <Typography variant="h6" component="h3" gutterBottom>
                Sources ({result.sources.length})
              </Typography>
              {result.sources.map((source, index) => (
                <Card key={index} sx={{ mb: 2 }}>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Source {index + 1}
                      </Typography>
                      <Chip 
                        label={`Score: ${source.score.toFixed(3)}`}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    </Box>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      {source.text}
                    </Typography>
                    {source.metadata && Object.keys(source.metadata).length > 0 && (
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Metadata: {JSON.stringify(source.metadata)}
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              ))}
            </>
          )}
        </Box>
      )}
    </Paper>
  );
}