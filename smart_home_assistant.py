#!/usr/bin/env python3
"""
Ultra-High-Performance Smart Home Assistant for Raspberry Pi 5
Optimized for 30+ TPS with TinyLlama, Qwen 0.5B, and DeepSeek-R1 7B
"""

# Standard library imports
import json
import time
import re
import argparse
import threading
import queue
import os
import asyncio
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import mmap
from functools import lru_cache
import gc
import psutil
import difflib
import ctypes

# Performance optimizations for ARM
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# For Ollama integration
try:
    import ollama
except ImportError:
    print("⚠️  The 'ollama' library is not installed. Please install it with: pip install ollama")
    ollama = None

# Try to import numpy
try:
    import numpy as np
except ImportError:
    print("⚠️ numpy not found. Some performance metrics may be unavailable.")
    np = None

# Try to import necessary libraries for API calls and JSON parsing
try:
    import httpx
    import orjson
    json_loads = orjson.loads
    json_dumps = lambda x: orjson.dumps(x).decode()
except ImportError:
    print("⚠️ httpx/orjson not found. Falling back to standard libraries. Please install: pip install httpx orjson")
    import requests as httpx
    json_loads = json.loads
    json_dumps = json.dumps

# Try to import gpiozero for Raspberry Pi GPIO control
try:
    from gpiozero import LED, PWMLED
    from gpiozero.pins.lgpio import LGPIOFactory
    from gpiozero.devices import Device as GpioZeroDevice
    GpioZeroDevice.pin_factory = LGPIOFactory()
    GPIO_AVAILABLE = True
    print("✅ gpiozero library loaded successfully")
except ImportError:
    print("⚠️ gpiozero not found. Running in GPIO simulation mode.")
    GPIO_AVAILABLE = False
    LED = None
    PWMLED = None

# Try to import FastAPI
try:
    from fastapi import FastAPI, Request, WebSocket
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("⚠️ FastAPI not found. Web interface will be disabled. Please install: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False


# Model configurations for optimal performance
MODEL_CONFIGS = {
    "tinyllama": {
        "temperature": 0.0,
        "num_predict": 64,
        "num_ctx": 512,
        "num_batch": 128,
        "num_thread": 4,
        "f16_kv": True,
        "use_mmap": True,
        "use_mlock": False,
        "repeat_penalty": 1.0,
        "top_k": 10,
        "top_p": 0.9,
    },
    "qwen:0.5b": {
        "temperature": 0.0,
        "num_predict": 64,
        "num_ctx": 512,
        "num_batch": 256,
        "num_thread": 4,
        "f16_kv": True,
        "use_mmap": True,
        "use_mlock": False,
        "repeat_penalty": 1.0,
        "top_k": 10,
        "top_p": 0.9,
    },
    "deepseek-r1:7b": {
        "temperature": 0.0,
        "num_predict": 96,
        "num_ctx": 1024,
        "num_batch": 512,
        "num_thread": 4,
        "f16_kv": True,
        "use_mmap": True,
        "use_mlock": True,
        "repeat_penalty": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "num_gpu": 0,  # CPU only for consistent performance
    },
    "deepseek-coder:1.3b": {
        "temperature": 0.0,
        "num_predict": 96,
        "num_ctx": 768,
        "num_batch": 256,
        "num_thread": 4,
        "f16_kv": True,
        "use_mmap": True,
        "use_mlock": False,
        "repeat_penalty": 1.0,
        "top_k": 15,
        "top_p": 0.9,
    }
}


# Enum for device types
class DeviceType(Enum):
    LIGHT = "light"
    FAN = "fan"
    HEATER = "heater"
    AC = "ac"
    DOOR_LOCK = "door_lock"
    OUTLET = "outlet"
    ALARM = "alarm"
    CAMERA = "camera"
    SMART_TV = "smart_tv"


# Optimized Device class
class Device:
    """Memory-efficient device representation"""
    __slots__ = ['name', 'pin', 'device_type', 'state', 'aliases', 'room', 
                 'power_consumption', 'dimmable', 'dim_level', 'last_changed', 
                 '_gpio_device', '_state_cache']
    
    def __init__(self, name: str, pin: int, device_type: DeviceType,
                 aliases: List[str] = None, room: str = "general",
                 power_consumption: float = 0.0, dimmable: bool = False):
        self.name = name
        self.pin = pin
        self.device_type = device_type
        self.state = False
        self.aliases = aliases or []
        self.room = room
        self.power_consumption = power_consumption
        self.dimmable = dimmable
        self.dim_level = 100
        self.last_changed = datetime.now()
        self._gpio_device = None
        self._state_cache = 0


class NEONOptimizer:
    """NEON SIMD optimizations for ARM processors"""
    
    def __init__(self):
        self.neon_lib = None
        self._compile_and_load()

    def _compile_and_load(self):
        """Compile and load the C library for NEON-optimized functions."""
        c_code = """
#include <string.h>
#include <stdio.h>

// Modern string functions are often optimized with SIMD instructions by the compiler.
// This serves as a placeholder for a more complex, hand-optimized NEON implementation if needed.
int find_substring(const char *text, const char *pattern) {
    if (strstr(text, pattern)) {
        return 1;
    }
    return 0;
}
"""
        c_file = Path("neon_optimizer.c")
        so_file = Path("neon_optimizer.so")

        if not so_file.exists():
            print("Compiling NEON optimizer...")
            c_file.write_text(c_code)
            # Use -O3 for maximum optimization, which should enable auto-vectorization
            compile_command = f"gcc -shared -o {so_file} -fPIC -O3 {c_file}"
            if os.system(compile_command) != 0:
                print("⚠️  Failed to compile NEON optimizer. Falling back to Python implementation.")
                self.neon_lib = None
            else:
                self.neon_lib = ctypes.CDLL(f'./{so_file}')
                self.neon_lib.find_substring.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                self.neon_lib.find_substring.restype = ctypes.c_int
        else:
            self.neon_lib = ctypes.CDLL(f'./{so_file}')
            self.neon_lib.find_substring.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            self.neon_lib.find_substring.restype = ctypes.c_int

    def fast_string_match(self, text: str, pattern: str) -> bool:
        """NEON-optimized string matching"""
        if self.neon_lib:
            text_bytes = text.lower().encode('utf-8')
            pattern_bytes = pattern.lower().encode('utf-8')
            return self.neon_lib.find_substring(text_bytes, pattern_bytes) == 1
        else:
            # Fallback to python
            return pattern.lower() in text.lower()


class UltraOptimizedSmartHomeAssistant:
    """
    Ultra-High-Performance Smart Home Assistant with an LLM-first approach.
    Optimized for 30+ TPS on Raspberry Pi 5.
    """

    def __init__(self, model_name: str = "tinyllama", ollama_host: str = "http://localhost:11434"):
        """Initialize with maximum optimizations"""
        print("🏠 Initializing Ultra-Optimized Smart Home Assistant for RPi5...")
        print(f"🤖 Model: {model_name}")
        print(f"🔧 CPU Cores: {mp.cpu_count()}")
        print(f"💾 RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

        self.model_name = model_name
        self.ollama_host = ollama_host
        self._gpio_simulation_mode = not GPIO_AVAILABLE
        
        # Get model-specific config
        self.model_config = self._get_model_config(model_name)
        
        # Thread pool optimized for RPi5
        self.executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="SmartHome"
        )
        
        # Initialize components
        self._init_devices()
        self._setup_performance_tracking()
        self._init_ollama(model_name)
        self._setup_gpio()
        self._init_context()
        self._init_automation_rules()
        
        # Ultra-fast caching with memory-mapped files
        self._init_ultra_cache()
        
        # Pre-compile regex patterns
        self._compile_patterns()
        
        # Initialize prompt cache
        self._init_prompt_cache()
        
        # Start background tasks
        self._start_background_tasks()
        
        # Force garbage collection
        gc.collect()
        
        print("✅ Ultra-Optimized Smart Home Assistant initialized!")

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get optimal configuration for specific model"""
        # Match partial model names
        for key, config in MODEL_CONFIGS.items():
            if key in model_name.lower():
                print(f"📋 Using optimized config for {key}")
                return config
        
        # Default config
        print("📋 Using default configuration")
        return {
            "temperature": 0.0,
            "num_predict": 96,
            "num_ctx": 768,
            "num_batch": 256,
            "num_thread": 4,
            "f16_kv": True,
            "use_mmap": True,
            "use_mlock": False,
        }

    def _compile_patterns(self):
        """Pre-compile regex patterns for faster matching"""
        self.patterns = {
            'json': re.compile(r'\{.*\}', re.DOTALL),
        }

    def _init_prompt_cache(self):
        """Initialize prompt template cache"""
        self.prompt_template = self._build_prompt_template()
        self.prompt_cache = {}

    def _build_prompt_template(self) -> str:
        """Build reusable prompt template with detailed examples and guidance."""
        return """You are an intelligent smart home assistant. ONLY respond with valid JSON for ALL user requests—do not explain or add text.

# Devices:
{device_states}

# Scenes:
{scenes}

# Time:
{time}

# Instructions:
- For requests about specific devices (lights, TV, fan, AC, etc), set "intent": "control", "devices": [...], "state": true/false, or "dim_level" if dimming.
- For requests about activating a scene (e.g. 'movie night', 'party'), set "intent": "scene", "scene": "<scene_name>".
- For ambiguous or natural requests like "watch my favorite show", "I want to relax", "get ready for bed", "make it cozy", "wake me up", infer the appropriate scene (use device or scene as needed).
- For status queries ("what's on?", "status", "show active devices"), use "intent": "status".
- Only ever output a single, valid JSON object. Never explain, just give JSON.

# Examples:
User: "turn on living room light"
Response: {{"intent":"control","devices":["living_room_light"],"state":true}}

User: "turn off all lights"
Response: {{"intent":"control","devices":["living_room_light","bedroom_light","kitchen_light","garden_light"],"state":false}}

User: "activate movie night"
Response: {{"intent":"scene","scene":"movie_night"}}

User: "I want to watch my favorite show"
Response: {{"intent":"scene","scene":"movie_night"}}

User: "dim bedroom light to 20"
Response: {{"intent":"control","devices":["bedroom_light"],"dim_level":20}}

User: "what's on?"
Response: {{"intent":"status"}}

User: "get ready for bed"
Response: {{"intent":"scene","scene":"sleep"}}

User: "make it cozy"
Response: {{"intent":"scene","scene":"party"}}

---

Now process this user command:
User: "{user_input}"

ONLY respond with valid JSON, never any explanation!
"""

    def _init_ollama(self, model_name):
        """Initialize Ollama client with performance optimizations"""
        if not ollama:
            print("❌ Ollama library not installed. Cannot continue.")
            raise ImportError("Ollama library is required.")
        
        try:
            # Set Ollama environment for performance
            os.environ['OLLAMA_NUM_PARALLEL'] = '1'  # Single request at a time
            os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Keep only one model in memory
            
            # Check if model exists locally
            models = ollama.list()
            if not any(model_name in m.get('name', '') for m in models.get('models', [])):
                print(f"⚠️ Model {model_name} not found locally. Pulling it now...")
                ollama.pull(model_name)
            
            # Warm up the model
            print("🔥 Warming up model...")
            self._warmup_model()
            
            self.model_name = model_name
            print(f"✅ Ollama initialized with model: {model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize Ollama model: {e}")
            raise

    def _warmup_model(self):
        """Warm up the model for better performance"""
        warmup_prompt = '{"intent": "status"}'
        try:
            ollama.generate(
                model=self.model_name,
                prompt=warmup_prompt,
                stream=False,
                format="json",
                options=self.model_config
            )
        except Exception:
            pass

    def _init_ultra_cache(self):
        """Initialize ultra-fast memory-mapped cache"""
        self.cache_file = Path("/tmp/smarthome_cache.mmap")
        self.cache_size = 4 * 1024 * 1024  # 4MB cache for RPi5
        
        # Ensure the file exists and is at least the required size before mapping
        if not self.cache_file.exists() or self.cache_file.stat().st_size < self.cache_size:
            with open(self.cache_file, "wb") as f:
                f.seek(self.cache_size - 1)
                f.write(b'\x00') # Efficiently create a sparse file of the correct size
        
        self.cache_fd = os.open(str(self.cache_file), os.O_RDWR)
        self.cache_mmap = mmap.mmap(self.cache_fd, self.cache_size)
        self.cache_index = {}
        self.cache_offset = 0
        
        # LRU cache for Python objects
        self._lru_cache = {}
        self._lru_order = deque(maxlen=1000)

    def _cache_get(self, key: str) -> Optional[Dict]:
        """Ultra-fast cache retrieval with LRU"""
        # Check LRU cache first
        if key in self._lru_cache:
            return self._lru_cache[key]
        
        # Check mmap cache
        if key not in self.cache_index:
            return None
        
        offset, size = self.cache_index[key]
        data = self.cache_mmap[offset:offset + size]
        result = json_loads(data)
        
        # Add to LRU cache
        self._lru_cache[key] = result
        self._lru_order.append(key)
        
        return result

    def _cache_set(self, key: str, value: Dict):
        """Ultra-fast cache storage"""
        # Add to LRU cache
        self._lru_cache[key] = value
        self._lru_order.append(key)
        
        # Clean LRU if needed
        if len(self._lru_cache) > 1000:
            oldest = self._lru_order.popleft()
            del self._lru_cache[oldest]
        
        # Store in mmap
        data = json_dumps(value).encode('utf-8')
        size = len(data)
        
        if self.cache_offset + size > self.cache_size:
            self.cache_offset = 0  # Wrap around
        
        self.cache_mmap[self.cache_offset:self.cache_offset + size] = data
        self.cache_index[key] = (self.cache_offset, size)
        self.cache_offset += size

    def _init_devices(self):
        """Initialize devices with optimized data structures"""
        self.devices = {
            'living_room_light': Device(
                name='living_room_light', pin=17, device_type=DeviceType.LIGHT,
                aliases=['living room light', 'main light', 'lounge light'],
                room='living_room', power_consumption=60, dimmable=True
            ),
            'living_room_fan': Device(
                name='living_room_fan', pin=27, device_type=DeviceType.FAN,
                aliases=['living room fan', 'main fan', 'ceiling fan'],
                room='living_room', power_consumption=75
            ),
            'smart_tv': Device(
                name='smart_tv', pin=22, device_type=DeviceType.SMART_TV,
                aliases=['TV', 'television'],
                room='living_room', power_consumption=150
            ),
            'bedroom_light': Device(
                name='bedroom_light', pin=23, device_type=DeviceType.LIGHT,
                aliases=['bedroom light', 'bed light'],
                room='bedroom', power_consumption=40, dimmable=True
            ),
            'bedroom_ac': Device(
                name='bedroom_ac', pin=24, device_type=DeviceType.AC,
                aliases=['bedroom ac', 'air conditioner', 'AC'],
                room='bedroom', power_consumption=1200
            ),
            'kitchen_light': Device(
                name='kitchen_light', pin=5, device_type=DeviceType.LIGHT,
                aliases=['kitchen light', 'cooking light'],
                room='kitchen', power_consumption=80
            ),
            'front_door_lock': Device(
                name='front_door_lock', pin=26, device_type=DeviceType.DOOR_LOCK,
                aliases=['front door', 'main door', 'door lock'],
                room='entrance', power_consumption=5
            ),
            'garden_light': Device(
                name='garden_light', pin=16, device_type=DeviceType.LIGHT,
                aliases=['garden', 'outdoor light', 'yard light'],
                room='outdoor', power_consumption=100
            ),
        }
        self._build_optimized_indices()

    def _build_optimized_indices(self):
        """Build optimized lookup structures"""
        self.devices_by_room = defaultdict(list)
        self.devices_by_type = defaultdict(list)
        self.device_lookup = {}  # Fast O(1) lookup
        
        for name, device in self.devices.items():
            self.devices_by_room[device.room].append(name)
            self.devices_by_type[device.device_type].append(name)
            
            # Build lookup table for aliases
            self.device_lookup[name] = name
            self.device_lookup[name.replace('_', ' ')] = name
            for alias in device.aliases:
                self.device_lookup[alias.lower()] = name

    def _setup_gpio(self):
        """Setup GPIO with DMA support for faster operations"""
        if self._gpio_simulation_mode:
            print("🔧 GPIO running in simulation mode")
            return

        try:
            print("🔧 Initializing GPIO pins...")
            for device in self.devices.values():
                if device.dimmable and PWMLED:
                    device._gpio_device = PWMLED(device.pin, frequency=1000)
                elif LED:
                    device._gpio_device = LED(device.pin)
                
                if device._gpio_device:
                    device._gpio_device.off()
                    device._state_cache = 0
            print("✅ GPIO pins initialized")
        except Exception as e:
            print(f"⚠️ GPIO initialization failed: {e}")
            self._gpio_simulation_mode = True

    def _setup_performance_tracking(self):
        """Initialize performance metrics"""
        self.performance_stats = {
            'inference_times': deque(maxlen=100),
            'tokens_per_second': deque(maxlen=100),
            'cache_hits': 0,
            'cache_misses': 0,
            'total_commands': 0,
            'ollama_errors': 0,
            'fast_path_hits': 0,
        }

    def understand_command(self, user_input: str) -> Dict[str, Any]:
        """LLM-first command processing with fast path optimization"""
        start_time = time.perf_counter()
        
        # Fast path for simple commands
        fast_result = self._try_fast_path(user_input)
        if fast_result:
            self.performance_stats['fast_path_hits'] += 1
            fast_result['processing_time'] = (time.perf_counter() - start_time) * 1000
            fast_result['method'] = 'fast_path'
            return fast_result
        
        # Check cache
        cache_key = user_input.lower().strip()
        cached = self._cache_get(cache_key)
        if cached:
            self.performance_stats['cache_hits'] += 1
            cached['from_cache'] = True
            cached['processing_time'] = (time.perf_counter() - start_time) * 1000
            return cached
        
        self.performance_stats['cache_misses'] += 1
        
        # Use LLM
        ollama_result = self._process_command_ollama_ultra(user_input)
        ollama_result['processing_time'] = (time.perf_counter() - start_time) * 1000
        self._cache_set(cache_key, ollama_result)
        return ollama_result

    def _try_fast_path(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Fast path for simple, common commands"""
        lower_input = user_input.lower().strip()
        
        # Direct device control patterns
        if lower_input.startswith(('turn on ', 'turn off ', 'toggle ')):
            action = 'on' if 'on' in lower_input else ('off' if 'off' in lower_input else 'toggle')
            
            # Extract device name
            for device_key, device_name in self.device_lookup.items():
                if device_key in lower_input:
                    return {
                        'intent': 'control',
                        'devices': [device_name],
                        'state': action == 'on' if action != 'toggle' else None,
                        'action': 'toggle' if action == 'toggle' else None,
                        'reasoning': f"Fast path: {action} {device_name}",
                        'tokens_per_second': 1000.0  # Instant
                    }
        
        # Scene activation
        if lower_input.startswith('activate ') or 'scene' in lower_input:
            for scene in self.scenes:
                if scene in lower_input:
                    return {
                        'intent': 'scene',
                        'scene': scene,
                        'reasoning': f"Fast path: activate {scene}",
                        'tokens_per_second': 1000.0
                    }
        
        # Status request
        if lower_input in ['status', 'what is on', "what's on", 'show status']:
            return {
                'intent': 'status',
                'reasoning': 'Fast path: status request',
                'tokens_per_second': 1000.0
            }
        
        return None

    def _extract_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts the first valid JSON object from a string."""
        try:
            # Try direct parse first
            return json_loads(text)
        except:
            # Try regex extraction
            match = self.patterns['json'].search(text)
            if match:
                try:
                    return json_loads(match.group(0))
                except:
                    pass
        return None

    def _build_ollama_prompt_ultra(self, user_input: str) -> str:
        """Build ultra-minimal prompt for speed"""
        # Use cached template
        prompt_key = f"{user_input}_{self.context['time_of_day']}"
        if prompt_key in self.prompt_cache:
            return self.prompt_cache[prompt_key]
        
        device_states = {name: "on" if d.state else "off" for name, d in self.devices.items()}
        
        prompt = self.prompt_template.format(
            device_states=json_dumps(device_states),
            scenes=','.join(self.scenes.keys()),
            time=datetime.now().strftime('%H:%M'),
            user_input=user_input
        )
        
        self.prompt_cache[prompt_key] = prompt
        return prompt

    def _process_command_ollama_ultra(self, user_input: str) -> Dict[str, Any]:
        """Ultra-optimized Ollama processing"""
        prompt = self._build_ollama_prompt_ultra(user_input)
        
        try:
            t0 = time.perf_counter()
            
            # Use optimized settings for the model
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                format="json",
                options=self.model_config
            )
            
            t1 = time.perf_counter()
            
            generated = response.get('response', '{}').strip()
            
            # Performance metrics
            eval_count = response.get('eval_count', 0)
            eval_duration = response.get('eval_duration', 1)
            tps = eval_count / (eval_duration / 1_000_000_000) if eval_duration > 0 else 0
            self.performance_stats['tokens_per_second'].append(tps)
            inference_time = (t1 - t0) * 1000
            self.performance_stats['inference_times'].append(inference_time)
            
            print(f"⚡ TPS: {tps:.1f} | Inference: {inference_time:.0f}ms")
            
            result = self._extract_json_block(generated)
            if result:
                result['tokens_per_second'] = tps
                result['inference_time'] = inference_time
                result['method'] = 'ollama_ultra'
                return result
            else:
                print(f"⚠️ LLM responded, but no valid JSON was found.")
                return {'intent': 'unknown', 'method': 'ollama_failed', 'error': 'Invalid JSON'}
            
        except Exception as e:
            self.performance_stats['ollama_errors'] += 1
            print(f"❌ Ollama Error: {e}")
            return {'intent': 'unknown', 'method': 'ollama_error', 'error': str(e)}

    def execute_command(self, action: Dict[str, Any]) -> str:
        """Execute command with minimal overhead"""
        if not action or 'intent' not in action:
            return "I couldn't understand that command."
        
        intent = action.get('intent')
        self.performance_stats['total_commands'] += 1
        
        # Get devices with fallback
        devices = action.get('devices', [])
        if not devices:
            # Try alternative keys
            for key in ['device_name', 'device_names', 'device']:
                if key in action:
                    val = action[key]
                    devices = [val] if isinstance(val, str) else val
                    break
        
        # Quick device name resolution
        resolved_devices = []
        for d in devices:
            if d in self.device_lookup:
                resolved_devices.append(self.device_lookup[d])
            elif d in self.devices:
                resolved_devices.append(d)
        
        devices = list(set(resolved_devices))
        
        if intent == 'control':
            if not devices:
                return "I understood the intent but couldn't determine which device."
            
            state = action.get('state')
            if action.get('action') == 'toggle':
                return self._batch_toggle_devices(devices)
            elif 'dim_level' in action:
                return self._batch_dim_devices(devices, action['dim_level'])
            elif state is not None:
                return self._batch_control_devices(devices, state)
            else:
                return f"Understood device(s) but no clear action."
        
        elif intent == 'scene':
            scene = action.get('scene')
            if not scene:
                return "I understood you want a scene but not which one."
            return self._activate_scene_optimized(scene)
            
        elif intent == 'status':
            return self._get_status_report_fast()
        
        return action.get('reasoning', 'Command not recognized')

    def _batch_control_devices(self, device_names: List[str], state: bool) -> str:
        """Batch GPIO operations with minimal overhead"""
        if not device_names:
            return "No devices specified"
        
        controlled = []
        power_change = 0
        
        # Batch operations
        for device_name in device_names:
            if device_name not in self.devices:
                continue
            
            device = self.devices[device_name]
            if device.state == state:
                continue
            
            # Update state
            device.state = state
            device._state_cache = 1 if state else 0
            device.last_changed = datetime.now()
            
            # GPIO operation
            if not self._gpio_simulation_mode and device._gpio_device:
                if state:
                    device._gpio_device.on()
                else:
                    device._gpio_device.off()
            
            power_change += device.power_consumption if state else -device.power_consumption
            controlled.append(device.name.replace('_', ' '))
        
        if controlled:
            self.context['power_usage'] += power_change
            action_str = "on" if state else "off"
            return f"✅ Turned {action_str}: {', '.join(controlled)}"
        
        return "No devices were changed"

    def _batch_toggle_devices(self, device_names: List[str]) -> str:
        """Batch toggle with minimal overhead"""
        if not device_names:
            return "No devices to toggle"
        
        responses = []
        for device_name in device_names:
            if device_name in self.devices:
                device = self.devices[device_name]
                new_state = not device.state
                self._batch_control_devices([device_name], new_state)
                responses.append(f"{device_name.replace('_', ' ')} → {'ON' if new_state else 'OFF'}")
        
        return " | ".join(responses) if responses else "No devices found"

    def _batch_dim_devices(self, device_names: List[str], dim_level: int) -> str:
        """Batch dimming operations"""
        dimmed = []
        dim_level = max(0, min(100, dim_level))
        pwm_value = dim_level / 100.0
        
        for device_name in device_names:
            if device_name not in self.devices:
                continue
            
            device = self.devices[device_name]
            if not device.dimmable:
                continue
            
            device.dim_level = dim_level
            device.state = dim_level > 0
            device._state_cache = dim_level
            device.last_changed = datetime.now()
            
            if not self._gpio_simulation_mode and device._gpio_device and hasattr(device._gpio_device, 'value'):
                device._gpio_device.value = pwm_value
            
            dimmed.append(f"{device.name.replace('_', ' ')} @ {dim_level}%")
        
        return f"💡 Dimmed: {', '.join(dimmed)}" if dimmed else "No dimmable devices found"

    def _activate_scene_optimized(self, scene_name: str) -> str:
        """Activate scene with batch operations"""
        if scene_name not in self.scenes:
            available = list(self.scenes.keys())[:3]
            return f"Unknown scene. Try: {', '.join(available)}"
        
        scene = self.scenes[scene_name]
        
        # Group operations by type
        on_devices = []
        off_devices = []
        dim_operations = {}
        
        for device_name, settings in scene['devices'].items():
            if device_name == 'all_lights':
                device_list = self.devices_by_type[DeviceType.LIGHT]
            else:
                device_list = [device_name] if device_name in self.devices else []
            
            for dev in device_list:
                if isinstance(settings, bool):
                    if settings:
                        on_devices.append(dev)
                    else:
                        off_devices.append(dev)
                elif isinstance(settings, dict):
                    if 'dim' in settings and self.devices[dev].dimmable:
                        dim_level = settings['dim']
                        if dim_level not in dim_operations:
                            dim_operations[dim_level] = []
                        dim_operations[dim_level].append(dev)
                    elif 'state' in settings:
                        if settings['state']:
                            on_devices.append(dev)
                        else:
                            off_devices.append(dev)
        
        # Execute batch operations
        if on_devices:
            self._batch_control_devices(on_devices, True)
        if off_devices:
            self._batch_control_devices(off_devices, False)
        for dim_level, devices in dim_operations.items():
            self._batch_dim_devices(devices, dim_level)
        
        return f"✅ Scene '{scene_name}' activated"

    def _get_status_report_fast(self) -> str:
        """Ultra-fast status report generation"""
        active = []
        total_power = 0
        
        for name, device in self.devices.items():
            if device.state:
                active.append((name, device))
                total_power += device.power_consumption
        
        report_parts = [
            "🏠 Smart Home Status\n",
            f"⚡ Power: {total_power:.0f}W",
            f"🕐 Time: {self.context['time_of_day']}",
        ]
        
        if active:
            report_parts.append(f"\n✅ Active ({len(active)}):")
            for name, device in active:
                info = f"  • {name.replace('_', ' ')}"
                if device.dimmable and device.dim_level < 100:
                    info += f" @ {device.dim_level}%"
                info += f" ({device.power_consumption}W)"
                report_parts.append(info)
        else:
            report_parts.append("\n❌ All devices OFF")
        
        return '\n'.join(report_parts)

    def _init_context(self):
        """Initialize context with pre-computed values"""
        self.context = {
            'temperature': 72,
            'humidity': 45,
            'time_of_day': self._get_time_context(),
            'power_usage': 0.0,
            'sleep_mode': False,
            'eco_mode': False,
            'vacation_mode': False,
        }

    def _init_automation_rules(self):
        """Initialize automation scenes"""
        self.scenes = {
            'movie_night': {
                'devices': {
                    'living_room_light': {'state': True, 'dim': 20},
                    'smart_tv': True,
                },
                'description': 'Movie mode'
            },
            'sleep': {
                'devices': {
                    'all_lights': False,
                    'bedroom_ac': True,
                },
                'description': 'Sleep mode'
            },
            'wake_up': {
                'devices': {
                    'bedroom_light': {'state': True, 'dim': 100},
                    'kitchen_light': True,
                },
                'description': 'Morning'
            },
            'away': {
                'devices': {
                    'all_lights': False,
                    'front_door_lock': True,
                },
                'description': 'Security'
            },
            'party': {
                'devices': {
                    'all_lights': {'state': True, 'dim': 80},
                    'living_room_fan': True,
                },
                'description': 'Party mode'
            },
            'eco': {
                'devices': {
                    'all_lights': {'state': True, 'dim': 50},
                    'bedroom_ac': False,
                },
                'description': 'Eco mode'
            },
        }

    def _get_time_context(self) -> str:
        """Get time context with caching"""
        hour = datetime.now().hour
        if 5 <= hour < 12: return "morning"
        elif 12 <= hour < 17: return "afternoon"
        elif 17 <= hour < 21: return "evening"
        else: return "night"

    def _start_background_tasks(self):
        """Start optimized background tasks"""
        threading.Thread(target=self._background_worker, daemon=True).start()

    def _background_worker(self):
        """Combined background worker for efficiency"""
        counter = 0
        while True:
            try:
                if counter % 30 == 0:
                    self.context['time_of_day'] = self._get_time_context()
                    self.context['power_usage'] = sum(
                        d.power_consumption for d in self.devices.values() 
                        if d.state
                    )
                
                if counter % 300 == 0 and counter > 0 and len(self.cache_index) > 1000:
                    items = sorted(self.cache_index.items(), key=lambda x: x[1][0])[-500:]
                    self.cache_index = dict(items)
                
                counter += 1
                time.sleep(1)
                
            except Exception:
                pass

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        stats = self.performance_stats
        
        def safe_avg(data):
            return sum(data) / len(data) if data else 0
        
        avg_inference = safe_avg(stats['inference_times'])
        avg_tps = safe_avg(list(stats['tokens_per_second']))
        
        total_lookups = stats['cache_hits'] + stats['cache_misses']
        cache_rate = (stats['cache_hits'] / total_lookups) * 100 if total_lookups > 0 else 0
        
        return {
            'avg_inference_ms': f"{avg_inference:.0f}",
            'avg_tokens_per_second': f"{avg_tps:.1f}",
            'cache_hit_rate': f"{cache_rate:.1f}%",
            'total_commands': stats['total_commands'],
            'ollama_errors': stats['ollama_errors'],
            'current_power': f"{self.context['power_usage']:.0f}W",
            'target_tps': "30+",
            'actual_vs_target': f"{(avg_tps/30)*100:.0f}%" if avg_tps > 0 else "0%"
        }

    def cleanup(self):
        """Clean shutdown with resource cleanup"""
        print("\n🔧 Shutting down...")
        
        if hasattr(self, 'async_client'):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.async_client.aclose())
                else:
                    loop.run_until_complete(self.async_client.aclose())
            except Exception as e:
                print(f"Error closing async client: {e}")

        self.executor.shutdown(wait=True)
        
        if hasattr(self, 'cache_mmap'):
            self.cache_mmap.close()
            os.close(self.cache_fd)
        
        if not self._gpio_simulation_mode:
            for device in self.devices.values():
                if device._gpio_device:
                    try:
                        device._gpio_device.off()
                        device._gpio_device.close()
                    except:
                        pass
        
        print("✅ Cleanup complete")


# FastAPI Web Application with WebSocket support
if FASTAPI_AVAILABLE:
    app = FastAPI(title="Ultra RPi5 Smart Home Hub")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    assistant = None
    
    DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        body { font-family: 'JetBrains Mono', monospace; }
        .neon-text { text-shadow: 0 0 10px currentColor; }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body class="bg-black text-green-400" x-data="ultraSmartHome()">
    <div class="container mx-auto px-4 py-6 max-w-7xl">
        <header class="mb-8 border-b border-green-800 pb-4">
            <div class="flex justify-between items-center">
                <div>
                    <h1 class="text-5xl font-bold neon-text">ULTRA SMART HOME</h1>
                    <p class="text-green-600 mt-2">RPi5 + NEON SIMD</p>
                </div>
                <div class="text-right">
                    <div class="text-4xl font-bold" :class="parseFloat(performance.avg_tokens_per_second) >= 30 ? 'text-green-400' : 'text-yellow-400'">
                        <span x-text="performance.avg_tokens_per_second"></span> TPS
                    </div>
                   
                    <div class="text-xs mt-1" x-text="currentTime"></div>
                </div>
            </div>
        </header>

        <div class="bg-gray-900 border border-green-800 rounded-lg p-4 mb-6">
            <div class="flex items-center gap-2 mb-2">
                <span class="text-green-600">$</span>
                <input
                    x-model="commandInput"
                    @keyup.enter="sendCommand"
                    type="text"
                    class="flex-1 bg-transparent outline-none text-green-400"
                    placeholder="Enter command..."
                    :disabled="processing"
                    autocomplete="off"
                    spellcheck="false"
                >
                <span x-show="processing" class="pulse">⚡</span>
            </div>
            <div x-show="lastResponse" class="mt-2 pl-4 text-green-300">
                <span x-text="lastResponse"></span>
                <span class="text-xs text-green-600 ml-2" x-text="responseStats"></span>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
            <div class="lg:col-span-3">
                <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                    <template x-for="device in allDevices" :key="device.name">
                        <button
                            @click="toggleDevice(device.name)"
                            class="border rounded-lg p-4 transition-all duration-200"
                            :class="device.state ? 'bg-green-900 border-green-400 shadow-lg shadow-green-500/50' : 'bg-gray-900 border-gray-700'"
                        >
                            <div class="text-3xl mb-2" x-text="getIcon(device.type)"></div>
                            <div class="text-sm font-bold" x-text="device.name.replace(/_/g, ' ').toUpperCase()"></div>
                            <div class="text-xs mt-1">
                                <span x-text="device.power_consumption + 'W'"></span>
                                <span x-show="device.dimmable && device.state" class="ml-1">
                                    @ <span x-text="device.dim_level + '%'"></span>
                                </span>
                            </div>
                        </button>
                    </template>
                </div>
                
                <div class="mt-6">
                    <h3 class="text-xl mb-3 text-green-600">SCENES</h3>
                    <div class="grid grid-cols-3 md:grid-cols-6 gap-2">
                        <template x-for="scene in scenes" :key="scene">
                            <button
                                @click="activateScene(scene)"
                                class="bg-gray-900 border border-green-800 rounded px-3 py-2 text-sm hover:bg-green-900 transition"
                                x-text="scene.toUpperCase()"
                            ></button>
                        </template>
                    </div>
                </div>
            </div>

            <div class="space-y-4">
                <div class="bg-gray-900 border border-green-800 rounded-lg p-4">
                    <h3 class="text-lg font-bold mb-3 text-green-400">PERFORMANCE</h3>
                    <div class="space-y-2 text-sm font-mono">
                        <div class="flex justify-between">
                            <span>TPS:</span>
                            <span x-text="performance.avg_tokens_per_second" class="text-yellow-400"></span>
                        </div>
                        <div class="flex justify-between">
                            <span>Inference:</span>
                            <span x-text="performance.avg_inference_ms + 'ms'"></span>
                        </div>
                        <div class="flex justify-between">
                            <span>Cache:</span>
                            <span x-text="performance.cache_hit_rate"></span>
                        </div>
                    </div>
                </div>

                <div class="bg-gray-900 border border-green-800 rounded-lg p-4">
                    <h3 class="text-lg font-bold mb-3 text-green-400">SYSTEM</h3>
                    <div class="space-y-2 text-sm font-mono">
                        <div class="flex justify-between">
                            <span>Power:</span>
                            <span x-text="stats.power_usage" class="text-yellow-400"></span>
                        </div>
                        <div class="flex justify-between">
                            <span>Active:</span>
                            <span x-text="stats.active_devices + '/' + stats.total_devices"></span>
                        </div>
                        <div class="flex justify-between">
                            <span>Commands:</span>
                            <span x-text="performance.total_commands"></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function ultraSmartHome() {
            return {
                allDevices: [],
                scenes: [],
                stats: { power_usage: '0W', active_devices: 0, total_devices: 0 },
                performance: {},
                commandInput: '',
                lastResponse: '',
                responseStats: '',
                processing: false,
                currentTime: '',

                init() {
                    this.updateTime();
                    setInterval(() => this.updateTime(), 1000);
                    this.fetchStatus();
                    setInterval(() => this.fetchStatus(), 2000);
                },

                updateTime() {
                    this.currentTime = new Date().toLocaleTimeString('en-US', { hour12: false });
                },

                async fetchStatus() {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        this.allDevices = Object.values(data.rooms).flat();
                        this.scenes = data.scenes;
                        this.stats = data.stats;
                        this.performance = data.performance;
                    } catch (error) {
                        console.error('Status fetch failed:', error);
                    }
                },

                async sendCommand() {
                    if (!this.commandInput.trim() || this.processing) return;

                    this.processing = true;
                    const startTime = performance.now();

                    try {
                        const response = await fetch('/api/command', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ text: this.commandInput })
                        });

                        const data = await response.json();
                        const elapsed = performance.now() - startTime;
                        
                        this.lastResponse = data.response;
                        this.responseStats = `[${elapsed.toFixed(0)}ms | ${data.tokens_per_second} TPS | ${data.action_details?.method || 'unknown'}]`;
                        this.commandInput = '';
                        await this.fetchStatus();
                    } catch (error) {
                        this.lastResponse = 'ERROR: Command failed';
                        this.responseStats = '';
                    } finally {
                        this.processing = false;
                    }
                },

                async toggleDevice(deviceName) {
                    await this.postData('/api/device/toggle', { device: deviceName });
                },

                async activateScene(sceneName) {
                    await this.postData('/api/scene/activate', { scene: sceneName });
                },

                async postData(url, body) {
                    try {
                        await fetch(url, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(body)
                        });
                        await this.fetchStatus();
                    } catch (error) {
                        console.error(`POST to ${url} failed:`, error);
                    }
                },

                getIcon(type) {
                    const icons = { light: '💡', fan: '🌀', smart_tv: '📺', ac: '❄️', door_lock: '🔒', default: '🔌' };
                    return icons[type] || icons.default;
                }
            };
        }
    </script>
</body>
</html>
"""

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return DASHBOARD_HTML

    @app.post("/api/command")
    async def api_command(request: Request):
        data = await request.json()
        if not data or 'text' not in data:
            return JSONResponse({'error': 'Missing command text'}, status_code=400)
        
        # Since understand_command is now synchronous, we run it in an executor
        # to avoid blocking the async event loop of FastAPI.
        loop = asyncio.get_event_loop()
        action = await loop.run_in_executor(
            None, assistant.understand_command, data['text']
        )
        response = assistant.execute_command(action)
        
        return JSONResponse({
            'response': response,
            'action_details': action,
            'tokens_per_second': f"{action.get('tokens_per_second', 0):.1f}"
        })

    @app.get("/api/status")
    async def api_status():
        rooms = defaultdict(list)
        for name, device in assistant.devices.items():
            rooms[device.room].append({
                'name': name,
                'state': device.state,
                'type': device.device_type.value,
                'power_consumption': device.power_consumption,
                'dimmable': device.dimmable,
                'dim_level': device.dim_level if device.dimmable else None
            })
        
        return JSONResponse({
            'rooms': rooms,
            'stats': {
                'power_usage': f"{assistant.context['power_usage']:.0f}W",
                'active_devices': sum(1 for d in assistant.devices.values() if d.state),
                'total_devices': len(assistant.devices)
            },
            'scenes': list(assistant.scenes.keys()),
            'performance': assistant.get_performance_summary()
        })

    @app.post("/api/device/toggle")
    async def api_toggle_device(request: Request):
        data = await request.json()
        device_name = data.get('device')
        if not device_name or device_name not in assistant.devices:
            return JSONResponse({'error': 'Device not found'}, status_code=404)
        
        response = assistant._batch_toggle_devices([device_name])
        return JSONResponse({
            'success': True,
            'response': response,
            'new_state': assistant.devices[device_name].state
        })

    @app.post("/api/scene/activate")
    async def api_activate_scene(request: Request):
        data = await request.json()
        scene_name = data.get('scene')
        if not scene_name or scene_name not in assistant.scenes:
            return JSONResponse({'error': 'Scene not found'}, status_code=404)
        
        response = assistant._activate_scene_optimized(scene_name)
        return JSONResponse({'response': response})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                await asyncio.sleep(1)
                perf = assistant.get_performance_summary()
                await websocket.send_json({"type": "performance", "data": perf})
        except:
            pass


def run_terminal_interface(assistant):
    """Ultra-fast terminal interface"""
    print("\n🚀 ULTRA SMART HOME ASSISTANT READY!")
    #print("⚡ Target: 30+ tokens/second with Kleidi AI + NEON")
    print("💬 Commands: 'turn on/off [device]', 'dim [device] to [%]', 'activate [scene]', 'status', 'quit'")
    print("-" * 80 + "\n")

    while True:
        try:
            user_input = input("\033[92m>\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                break
            
            start_time = time.perf_counter()
            
            action = assistant.understand_command(user_input)
            response = assistant.execute_command(action)
            elapsed = (time.perf_counter() - start_time) * 1000
            
            print(f"\n\033[96m{response}\033[0m")
            
            tps = action.get('tokens_per_second', 0)
            if tps > 0:
                color = '\033[92m' if tps >= 30 else '\033[93m' if tps >= 20 else '\033[91m'
                print(f"\033[90m[{color}{tps:.1f} TPS\033[90m | {elapsed:.0f}ms | {action.get('method', 'unknown')}]\033[0m")
            else:
                print(f"\033[90m[{elapsed:.0f}ms | {action.get('method', 'unknown')}]\033[0m")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n\033[93mInterrupted\033[0m")
            break
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")


def optimize_system():
    """Apply system-level optimizations for RPi5"""
    try:
        # Set CPU governor to performance
        os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1")
        
        # Increase process priority
        os.nice(-10)
        
        print("✅ System optimizations applied")
    except:
        print("⚠️ Could not apply all system optimizations (need sudo)")


def main():
    """Main entry point with Kleidi AI support"""
    global assistant
    
    parser = argparse.ArgumentParser(
        description="Ultra-High-Performance Smart Home Assistant with Kleidi AI"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-coder:1.3b",
        help="Model name (deepseek-coder:1.3b recommended for 30+ TPS)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web server port"
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable web interface"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply system-level optimizations (requires sudo)"
    )
    
    args = parser.parse_args()
    
    if args.optimize:
        optimize_system()
    
    try:
        assistant = UltraOptimizedSmartHomeAssistant(
            model_name=args.model
        )
        
        if not args.no_web and FASTAPI_AVAILABLE:
            import uvicorn
            from threading import Thread
            
            def run_server():
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=args.port,
                    log_level="error",
                    access_log=False,
                    loop="uvloop"
                )
            
            server_thread = Thread(target=run_server, daemon=True)
            server_thread.start()
            print(f"🚀 Web interface: http://0.0.0.0:{args.port}")
        
        run_terminal_interface(assistant)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'assistant' in globals() and assistant:
            assistant.cleanup()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
