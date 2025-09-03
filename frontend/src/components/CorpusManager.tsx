import React, { useState } from 'react';
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Box,
  Chip,
} from '@mui/material';
import { Delete, Add, Info } from '@mui/icons-material';
import { ragService } from '../services/ragService';
import { Corpus } from '../types';

interface CorpusManagerProps {
  corpora: Corpus[];
  onUpdate: () => void;
  selectedCorpus: string;
  onCorpusSelect: (corpusId: string) => void;
}

export default function CorpusManager({ 
  corpora, 
  onUpdate, 
  selectedCorpus,
  onCorpusSelect 
}: CorpusManagerProps) {
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newCorpusName, setNewCorpusName] = useState('');
  const [newCorpusDescription, setNewCorpusDescription] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleCreateCorpus = async () => {
    if (!newCorpusName.trim()) {
      setError('Corpus name is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await ragService.createCorpus(newCorpusName, newCorpusDescription);
      setCreateDialogOpen(false);
      setNewCorpusName('');
      setNewCorpusDescription('');
      onUpdate();
    } catch (err: any) {
      setError(err.message || 'Failed to create corpus');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteCorpus = async (corpusId: string) => {
    if (!window.confirm('Are you sure you want to delete this corpus? This action cannot be undone.')) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await ragService.deleteCorpus(corpusId);
      if (selectedCorpus === corpusId) {
        onCorpusSelect('');
      }
      onUpdate();
    } catch (err: any) {
      setError(err.message || 'Failed to delete corpus');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" component="h2">
          Corpus Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Create Corpus
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {corpora.length === 0 ? (
        <Typography variant="body1" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
          No corpora available. Create one to get started.
        </Typography>
      ) : (
        <List>
          {corpora.map((corpus) => (
            <ListItem 
              key={corpus.id}
              button
              selected={selectedCorpus === corpus.id}
              onClick={() => onCorpusSelect(corpus.id)}
              sx={{ 
                mb: 1, 
                border: 1, 
                borderColor: 'divider',
                borderRadius: 1,
              }}
            >
              <ListItemText
                primary={
                  <Box display="flex" alignItems="center" gap={1}>
                    {corpus.name}
                    {corpus.document_count > 0 && (
                      <Chip 
                        label={`${corpus.document_count} docs`} 
                        size="small" 
                        color="primary" 
                      />
                    )}
                  </Box>
                }
                secondary={corpus.description || 'No description'}
              />
              <ListItemSecondaryAction>
                <IconButton
                  edge="end"
                  aria-label="info"
                  onClick={(e) => {
                    e.stopPropagation();
                    // Could show more details in a dialog
                  }}
                >
                  <Info />
                </IconButton>
                <IconButton
                  edge="end"
                  aria-label="delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteCorpus(corpus.id);
                  }}
                  disabled={loading}
                >
                  <Delete />
                </IconButton>
              </ListItemSecondaryAction>
            </ListItem>
          ))}
        </List>
      )}

      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Corpus</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Corpus Name"
            fullWidth
            variant="outlined"
            value={newCorpusName}
            onChange={(e) => setNewCorpusName(e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Description (optional)"
            fullWidth
            multiline
            rows={3}
            variant="outlined"
            value={newCorpusDescription}
            onChange={(e) => setNewCorpusDescription(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)} disabled={loading}>
            Cancel
          </Button>
          <Button onClick={handleCreateCorpus} variant="contained" disabled={loading}>
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
}