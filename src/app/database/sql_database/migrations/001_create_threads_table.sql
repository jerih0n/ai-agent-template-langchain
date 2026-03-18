-- Migration: Create threads table
-- Description: Creates a table to store thread information for conversation management

CREATE TABLE IF NOT EXISTS threads (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary TEXT
);

-- Create index on thread_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_threads_thread_id ON threads(thread_id);
