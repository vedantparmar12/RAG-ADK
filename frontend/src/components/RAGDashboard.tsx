import React, { useState } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Tab,
  Tabs,
  CircularProgress,
  Alert,
} from '@mui/material';
import CorpusManager from './CorpusManager';
import DocumentUploader from './DocumentUploader';
import QueryInterface from './QueryInterface';
import { useQuery } from '@tanstack/react-query';
import { ragService } from '../services/ragService';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default function RAGDashboard() {
  const [tabValue, setTabValue] = useState(0);
  const [selectedCorpus, setSelectedCorpus] = useState<string>('');

  const { data: corpora, isLoading, error, refetch } = useQuery({
    queryKey: ['corpora'],
    queryFn: ragService.listCorpora,
    refetchInterval: 30000,
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">Failed to load corpora. Please check the backend connection.</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        RAG Agent Dashboard
      </Typography>
      
      <Paper sx={{ p: 2, mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="dashboard tabs">
          <Tab label="Query" />
          <Tab label="Corpus Management" />
          <Tab label="Document Upload" />
        </Tabs>
      </Paper>

      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <QueryInterface 
              corpora={corpora || []}
              selectedCorpus={selectedCorpus}
              onCorpusSelect={setSelectedCorpus}
            />
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <CorpusManager 
              corpora={corpora || []}
              onUpdate={refetch}
              selectedCorpus={selectedCorpus}
              onCorpusSelect={setSelectedCorpus}
            />
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <DocumentUploader 
              corpora={corpora || []}
              selectedCorpus={selectedCorpus}
              onCorpusSelect={setSelectedCorpus}
              onUploadComplete={refetch}
            />
          </Grid>
        </Grid>
      </TabPanel>
    </Container>
  );
}