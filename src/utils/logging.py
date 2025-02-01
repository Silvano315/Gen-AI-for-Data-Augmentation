import logging
import json
import sys
from pathlib import Path
from typing import Union, Dict, Any
from logging.handlers import RotatingFileHandler
from datetime import datetime


class GANLogger:
    """
    Custom logger for GAN training with both file and console output.
    
    Features:
    - JSON formatted logs
    - File rotation
    - Console output with colors
    - Different log levels for file and console
    - Metric tracking and persistence
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Union[str, Path],
        max_file_size: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name = name

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        
        file_handler = RotatingFileHandler(
            self.log_dir / f"{name}.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.metrics: Dict[str, list] = {}
        self.best_metrics: Dict[str, float] = {}
        
        self.info(f"Logger initialized in {self.log_dir}")

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics dictionary as a JSON string."""
        return json.dumps(metrics, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics and update tracking."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
            
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = value
                self.info(f"New best {key}: {value:.4f}")
        
        self.info(
            f"Step {step} - Metrics:\n{self._format_metrics(metrics)}"
        )

    def save_metrics(self):
        """Save metrics history to JSON file."""
        metrics_file = self.log_dir / f"{self.name}_metrics.json"
        metrics_data = {
            "history": self.metrics,
            "best": self.best_metrics,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.info(f"Metrics saved to {metrics_file}")

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)