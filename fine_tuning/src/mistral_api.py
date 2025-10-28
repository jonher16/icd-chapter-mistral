"""
Mistral Fine-tuning API wrapper.

This module provides a clean interface to the Mistral Fine-tuning API,
including job creation, monitoring, and inference.
Uses official Mistral client with WandB integration for automatic metric logging.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from mistralai import Mistral
from mistralai.models import WandbIntegration

logger = logging.getLogger(__name__)


class MistralClient:
    """Wrapper for Mistral API including fine-tuning, inference, and file management using official Mistral client with WandB integration."""
    
    def __init__(self, api_key: Optional[str] = None, wandb_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Mistral API client.
        
        Args:
            api_key: Mistral API key. If None, reads from MISTRAL_API_KEY env var.
            wandb_config: W&B configuration dict from config.yaml.
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        # Initialize official Mistral client
        self.client = Mistral(api_key=self.api_key)
        
        # Store W&B config as-is for job creation
        self.wandb_config = wandb_config or {}
        
        logger.info("Initialized Mistral API client")
    
    def upload_file(self, file_path: str, purpose: str = "fine-tune") -> str:
        """
        Upload a training file to Mistral.
        
        Args:
            file_path: Path to the JSONL file to upload.
            purpose: Purpose of the file (default: "fine-tune").
        
        Returns:
            File ID for use in fine-tuning job.
        """
        logger.info(f"Uploading file: {file_path}")
        
        with open(file_path, "rb") as f:
            try:
                result = self.client.files.upload(
                    file={
                        "file_name": Path(file_path).name,
                        "content": f,
                    },
                    purpose=purpose,
                )
            except Exception as e:
                logger.error(f"Failed to upload file {file_path}: {e}")
                raise RuntimeError("File upload failed") from e
        
        file_id = result.id
        logger.info(f"File uploaded successfully. File ID: {file_id}")
        return file_id
    
    def create_fine_tuning_job(
        self,
        training_file_id: str,
        validation_file_id: Optional[str] = None,
        model: str = "open-mistral-7b",
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a fine-tuning job with WandB integration.
        
        Args:
            training_file_id: ID of uploaded training file.
            validation_file_id: ID of uploaded validation file (optional).
            model: Base model to fine-tune.
            hyperparameters: Training hyperparameters (epochs, learning_rate, etc.).
            suffix: Suffix for the fine-tuned model name.
            auto_start: If False, creates job but doesn't start training (allows manual review).
        
        Returns:
            Job response dict with job_id.
        """
        logger.info(f"Creating fine-tuning job for model: {model} (auto_start=False)")
        
        hp = hyperparameters or {}
        
        training_params = {}
        if hp.get("training_steps") is not None:
            training_params["training_steps"] = hp["training_steps"]
        if hp.get("epochs") is not None:
            training_params["epochs"] = hp["epochs"]
        if hp.get("learning_rate") is not None:
            training_params["learning_rate"] = hp["learning_rate"]
        if hp.get("weight_decay") is not None:
            training_params["weight_decay"] = hp["weight_decay"]
        if hp.get("warmup_fraction") is not None:
            training_params["warmup_fraction"] = hp["warmup_fraction"]
        if hp.get("gradient_clip_norm") is not None:
            training_params["gradient_clip_norm"] = hp["gradient_clip_norm"]
        
        integrations = []
        if self.wandb_config.get("enabled"):
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if not wandb_api_key:
                logger.warning("WANDB_API_KEY not found - W&B integration disabled")
            else:
                wandb_integration = WandbIntegration(
                    type="wandb",
                    project=self.wandb_config.get("project", "mistral-ft"),
                    api_key=wandb_api_key,
                )
                integrations.append(wandb_integration)
                logger.info(f"W&B integration enabled: project={self.wandb_config.get('project', 'mistral-ft')}")
        
        training_files = [{"file_id": training_file_id, "weight": 1}]
        
        validation_files = [validation_file_id] if validation_file_id else None
        
        try:
            response = self.client.fine_tuning.jobs.create(
                model=model,
                training_files=training_files,
                validation_files=validation_files,
                hyperparameters=training_params if training_params else None,  # Pass dict directly
                suffix=suffix,
                integrations=integrations if integrations else None,
                auto_start=False
            )
        except Exception as e:
            logger.error(f"Failed to create fine-tuning job: {e}")
            raise RuntimeError("Fine-tuning job creation failed") from e
            
        job_dict = {
            "id": response.id,
            "status": response.status,
            "model": response.model,
            "created_at": response.created_at,
            "hyperparameters": training_params
        }
        
        logger.info(f"Fine-tuning job created successfully (auto_start=False):")
        logger.info(f"  Job ID: {response.id}")
        logger.info(f"  Status: {response.status}")
        logger.info(f"  Review the job in Mistral console, then start it manually")

        return job_dict
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Run inference using a fine-tuned model.
        
        Args:
            model: Model ID (fine-tuned model).
            messages: Chat messages in OpenAI format.
            max_tokens: Maximum tokens to generate (default: 100 for JSON output).
            temperature: Sampling temperature (0 = deterministic).
            top_p: Nucleus sampling parameter (default: 1.0).
            response_format: Response format spec (e.g., {"type": "json_object"}).
        
        Returns:
            Generated text response.
        """
        response = self.client.chat.complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
        )
        return response.choices[0].message.content

    def inference(
        self,
        model: str,
        messages_list: List[List[Dict[str, str]]],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        response_format: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Run batch inference on multiple examples.
        
        Args:
            model: Model ID (fine-tuned model).
            messages_list: List of message sequences.
            max_tokens: Maximum tokens per generation (default: 100 for JSON output).
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter (default: 1.0).
            batch_size: Not used (kept for API compatibility).
            response_format: Response format spec (e.g., {"type": "json_object"}).
        
        Returns:
            List of generated responses.
        """
        logger.info(f"Running inference on {len(messages_list)} examples")
        
        responses = []
        for i, messages in enumerate(messages_list):
            try:
                response = self.chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    response_format=response_format,
                )
                responses.append(response)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(messages_list)} examples")
                
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                responses.append("")  # Empty response on error
        
        logger.info(f"Inference complete: {len(responses)} responses")
        return responses
