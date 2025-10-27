#!/usr/bin/env python3
"""
Persistent training runner for GRPO with computational efficiency.
Optimized for 80GB GPU and remote execution with laptop disconnection.
"""

import os
import sys
import subprocess
import signal
import logging
from pathlib import Path

def setup_logging():
    """Setup comprehensive logging for monitoring training progress."""
    log_dir = Path("./output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def setup_environment_for_80gb_gpu():
    """Setup optimal environment variables for 80GB H100 GPU."""
    logger = logging.getLogger(__name__)
    
    # Optimized settings for 80GB GPU
    env_settings = {
        # VLLM memory settings - utilize more of the 80GB
    "VLLM_MEM_UTIL_EVAL": "0.25",  # Extra headroom for evaluation engine
    "VLLM_MEM_UTIL_ROLLOUT": "0.35",  # Keep plenty of space for HF policy weights
    "VLLM_MAX_NUM_SEQS_EVAL": "160",  # Moderate eval concurrency
    "VLLM_MAX_NUM_SEQS_ROLLOUT": "128",  # Limit rollout engine concurrency
        "VLLM_MAX_MODEL_LEN_EVAL": "8192",  # Longer sequences
        "VLLM_MAX_MODEL_LEN_ROLLOUT": "4096",  # Longer sequences
        
        # Training settings optimized for larger GPU
    "ROLLOUT_BATCH_SIZE": "192",  # Reduce per-step memory pressure
        "MAX_TOKENS": "512",  # Longer responses
        
        # Backend settings
        "ROLLOUT_BACKEND": "vllm",  # Use vLLM for faster rollouts
        "USE_VLLM_EVAL": "1",  # Use vLLM for evaluation
        
        # Performance optimizations
    # Disable torch.compile to avoid cudagraph incompatibilities with PPO loop
    "TORCH_COMPILE": "0",
    "TORCH_COMPILE_MODE": "max-autotune",
        
        # CUDA optimizations
        "CUDA_LAUNCH_BLOCKING": "0",
        "TORCH_CUDNN_ALLOW_TF32": "1",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
        
        # Prevent hanging issues
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        
        # Use vLLM v1 for stability
        "VLLM_USE_V1": "1",
    }
    
    for key, value in env_settings.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")
    
    logger.info("Environment configured for 80GB GPU training")

def create_tmux_session(session_name="grpo_training"):
    """Create a tmux session for persistent training."""
    logger = logging.getLogger(__name__)
    
    # Check if tmux is available
    try:
        subprocess.run(["which", "tmux"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        logger.info("Installing tmux for persistent sessions...")
        subprocess.run(["apt", "update"], check=True)
        subprocess.run(["apt", "install", "-y", "tmux"], check=True)
    
    # Kill existing session if it exists
    try:
        subprocess.run(["tmux", "kill-session", "-t", session_name], 
                      capture_output=True, check=False)
    except:
        pass
    
    # Create new session
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name], check=True)
        logger.info(f"Created tmux session: {session_name}")
        return session_name
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create tmux session: {e}")
        return None

def run_training_in_tmux(session_name, alpha=0.05):
    """Run the training inside tmux session."""
    logger = logging.getLogger(__name__)
    
    # Training command
    training_cmd = f"cd /root/GRPO && python run_computational_efficiency.py --alpha {alpha}"
    
    # Send command to tmux session
    try:
        subprocess.run([
            "tmux", "send-keys", "-t", session_name, 
            training_cmd, "Enter"
        ], check=True)
        
        logger.info(f"Started training in tmux session '{session_name}'")
        logger.info(f"Command: {training_cmd}")
        logger.info(f"To monitor: tmux attach -t {session_name}")
        logger.info(f"To detach: Ctrl+B then D")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start training in tmux: {e}")
        return False

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger = logging.getLogger(__name__)
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main function to setup and run persistent training."""
    logger = setup_logging()
    setup_signal_handlers()
    
    logger.info("="*60)
    logger.info("GRPO PERSISTENT TRAINING SETUP")
    logger.info("="*60)
    logger.info("Optimized for 80GB H100 GPU")
    logger.info("Training will continue after laptop disconnection")
    
    # Setup environment
    setup_environment_for_80gb_gpu()
    
    # Create tmux session
    session_name = create_tmux_session()
    if not session_name:
        logger.error("Failed to create tmux session, running directly...")
        # Fallback to direct execution
        from run_computational_efficiency import run_computational_efficiency_training
        run_computational_efficiency_training(alpha=0.05)
        return
    
    # Start training in tmux
    success = run_training_in_tmux(session_name, alpha=0.05)
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("TRAINING STARTED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Session: {session_name}")
        logger.info("Monitoring commands:")
        logger.info(f"  tmux attach -t {session_name}    # Attach to session")
        logger.info(f"  tmux list-sessions               # List all sessions")
        logger.info(f"  tail -f output/logs/training.log # Follow logs")
        logger.info("\nTraining will continue running even if you close your laptop!")
        logger.info("To stop training: tmux kill-session -t grpo_training")
        
        # Optionally attach to the session immediately
        print("\nWould you like to attach to the training session now? (y/n): ", end="")
        response = input().strip().lower()
        if response in ['y', 'yes']:
            os.system(f"tmux attach -t {session_name}")
    else:
        logger.error("Failed to start training in tmux session")
        sys.exit(1)

if __name__ == "__main__":
    main()