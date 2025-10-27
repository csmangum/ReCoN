#!/usr/bin/env python3
"""
Syllable Template Learning App

A simplified tkinter application focused on understanding how syllable/phoneme 
templates work in the ReCoN system. This helps users get better intuition on 
how to make terminal scripts more accurate.

Features:
- Interactive syllable buttons
- Real-time template activation visualization
- Audio feature explanation
- Step-by-step learning mode
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import time
from typing import Dict, List, Any
import numpy as np

# Import ReCoN modules
import sys
import os
sys.path.append('/workspace')

from recon_core.compiler import compile_from_dict
from recon_core.engine import Engine
from recon_core.config import EngineConfig
from recon_core.enums import State, UnitType
from perception.audio_terminals import create_synthetic_audio_features


class SyllableLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Syllable Template Learning App")
        self.root.geometry("1000x700")
        
        # Load the network
        self.load_network()
        
        # Initialize UI
        self.setup_ui()
        
        # Current state
        self.current_step = 0
        self.selected_syllable = None
        
    def load_network(self):
        """Load the audio phrase recognition network."""
        try:
            import yaml
            with open('/workspace/scripts/engage_active_perception.yaml', 'r') as f:
                network_spec = yaml.safe_load(f)
            
            # Compile the network
            self.graph = compile_from_dict(network_spec)
            
            # Create engine
            config = EngineConfig()
            config.deterministic_order = True
            config.confirmation_ratio = 0.75
            config.sur_positive = 0.4
            config.ret_feedback_enabled = True
            
            self.engine = Engine(self.graph, config)
            
            # Extract phonemes (our "syllables")
            self.syllables = []
            for unit_id, unit in self.graph.units.items():
                if unit.kind == UnitType.SCRIPT and 'phoneme' in unit_id:
                    self.syllables.append(unit_id)
            
            # Get terminal features
            self.terminals = []
            for unit_id, unit in self.graph.units.items():
                if unit.kind == UnitType.TERMINAL:
                    self.terminals.append(unit_id)
            
            print(f"Loaded {len(self.syllables)} syllables and {len(self.terminals)} terminals")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load network: {e}")
            self.root.quit()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Syllable Template Learning App", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                               text="Click on syllable buttons to see how they activate template features.\n" +
                                    "This helps you understand how to make terminal scripts more accurate.",
                               font=('Arial', 10))
        instructions.pack(pady=(0, 10))
        
        # Create main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Syllable buttons
        left_panel = ttk.LabelFrame(content_frame, text="Syllables (Click to Activate)", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.setup_syllable_buttons(left_panel)
        
        # Right panel - Template visualization
        right_panel = ttk.LabelFrame(content_frame, text="Template Activation", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.setup_template_display(right_panel)
        
        # Bottom panel - Controls and info
        bottom_panel = ttk.Frame(main_frame)
        bottom_panel.pack(fill=tk.X, pady=(10, 0))
        
        self.setup_controls(bottom_panel)
    
    def setup_syllable_buttons(self, parent):
        """Set up the syllable buttons."""
        # Create scrollable frame for buttons
        canvas = tk.Canvas(parent, height=400)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create syllable buttons
        self.syllable_buttons = {}
        for i, syllable in enumerate(self.syllables):
            # Extract display name
            display_name = syllable.replace('u_', '').replace('_phoneme', '')
            
            # Create button
            btn = tk.Button(scrollable_frame, 
                          text=display_name,
                          command=lambda s=syllable: self.activate_syllable(s),
                          width=15, height=2,
                          font=('Arial', 10),
                          bg='lightgray')
            btn.pack(pady=2, fill=tk.X)
            
            self.syllable_buttons[syllable] = btn
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_template_display(self, parent):
        """Set up the template activation display."""
        # Create notebook for different views
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Template features tab
        features_frame = ttk.Frame(notebook)
        notebook.add(features_frame, text="Template Features")
        
        self.features_text = tk.Text(features_frame, height=15, width=50, font=('Courier', 9))
        features_scrollbar = ttk.Scrollbar(features_frame, orient=tk.VERTICAL, command=self.features_text.yview)
        self.features_text.configure(yscrollcommand=features_scrollbar.set)
        
        self.features_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        features_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Network state tab
        state_frame = ttk.Frame(notebook)
        notebook.add(state_frame, text="Network State")
        
        self.state_text = tk.Text(state_frame, height=15, width=50, font=('Courier', 9))
        state_scrollbar = ttk.Scrollbar(state_frame, orient=tk.VERTICAL, command=self.state_text.yview)
        self.state_text.configure(yscrollcommand=state_scrollbar.set)
        
        self.state_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        state_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Learning tips tab
        tips_frame = ttk.Frame(notebook)
        notebook.add(tips_frame, text="Learning Tips")
        
        tips_text = tk.Text(tips_frame, height=15, width=50, font=('Arial', 10), wrap=tk.WORD)
        tips_scrollbar = ttk.Scrollbar(tips_frame, orient=tk.VERTICAL, command=tips_text.yview)
        tips_text.configure(yscrollcommand=tips_scrollbar.set)
        
        tips_content = """
SYLLABLE TEMPLATE LEARNING TIPS:

1. UNDERSTANDING TEMPLATES:
   - Each syllable/phoneme has a "template" that defines what audio features it expects
   - Templates are made up of terminal units (MFCC, pitch, formant, etc.)
   - When you click a syllable, it activates its template and shows you what features are important

2. AUDIO FEATURES EXPLAINED:
   - t_mfcc_low: Low frequency features (good for vowels like /a/, /e/, /o/)
   - t_pitch_high: High frequency features (good for consonants like /s/, /t/, /k/)
   - t_formant: Vowel formant structure (crucial for distinguishing vowels)
   - t_rhythm: Timing and rhythm patterns
   - t_spectrogram: Overall frequency content
   - t_noise_level: Background noise (inhibits recognition when high)

3. MAKING TERMINALS MORE ACCURATE:
   - Vowels need strong MFCC and formant features
   - Consonants need strong pitch and spectrogram features
   - Adjust thresholds based on what you observe
   - Consider noise levels in real environments
   - Test with different syllable combinations

4. TEMPLATE ACTIVATION PATTERNS:
   - Green = CONFIRMED (template fully activated)
   - Yellow = ACTIVE (template partially activated)
   - Orange = REQUESTED (template waiting for evidence)
   - Gray = INACTIVE (template not activated)

5. EXPERIMENTATION:
   - Try clicking different syllables to see their unique patterns
   - Notice how similar syllables have similar feature patterns
   - Use this to understand why some syllables are confused
   - Adjust terminal weights and thresholds accordingly
        """
        
        tips_text.insert(tk.END, tips_content)
        tips_text.configure(state=tk.DISABLED)
        
        tips_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tips_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_controls(self, parent):
        """Set up control buttons."""
        # Control buttons
        ttk.Button(parent, text="Reset All", command=self.reset_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(parent, text="Show All Features", command=self.show_all_features).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(parent, text="Run Full Simulation", command=self.run_full_simulation).pack(side=tk.LEFT, padx=(0, 5))
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Click a syllable to begin")
        ttk.Label(parent, textvariable=self.status_var).pack(side=tk.RIGHT)
    
    def activate_syllable(self, syllable_id):
        """Activate a specific syllable and show its template effects."""
        self.selected_syllable = syllable_id
        self.status_var.set(f"Activating syllable: {syllable_id}")
        
        # Reset the network
        self.engine.reset()
        
        # Manually activate the syllable
        if syllable_id in self.graph.units:
            unit = self.graph.units[syllable_id]
            unit.state = State.ACTIVE
            unit.a = 1.0
            
            # Run a few steps to propagate the activation
            for _ in range(3):
                self.engine.step()
            
            # Update displays
            self.update_syllable_buttons()
            self.update_template_display()
            
            self.status_var.set(f"Activated {syllable_id} - Check template features")
    
    def reset_all(self):
        """Reset all syllables and displays."""
        self.engine.reset()
        self.selected_syllable = None
        self.update_syllable_buttons()
        self.update_template_display()
        self.status_var.set("Reset - Click a syllable to begin")
    
    def show_all_features(self):
        """Show all terminal features without activating any syllable."""
        self.engine.reset()
        self.selected_syllable = None
        self.update_syllable_buttons()
        self.update_template_display()
        self.status_var.set("Showing all features - Click a syllable to see specific activation")
    
    def run_full_simulation(self):
        """Run the full phrase recognition simulation."""
        self.engine.reset()
        self.selected_syllable = None
        
        # Run the simulation
        for step in range(10):
            self.engine.step()
            self.update_syllable_buttons()
            self.update_template_display()
            self.root.update()
            time.sleep(0.3)
        
        self.status_var.set("Full simulation completed - Check results")
    
    def update_syllable_buttons(self):
        """Update syllable button colors based on their states."""
        for syllable_id, button in self.syllable_buttons.items():
            if syllable_id in self.graph.units:
                unit = self.graph.units[syllable_id]
                state = unit.state
                activation = unit.a
                
                # Color coding
                if state == State.CONFIRMED:
                    color = 'lightgreen'
                elif state == State.ACTIVE:
                    color = 'yellow'
                elif state == State.REQUESTED:
                    color = 'orange'
                else:
                    color = 'lightgray'
                
                button.configure(bg=color)
                
                # Update text with activation level
                display_name = syllable_id.replace('u_', '').replace('_phoneme', '')
                button.configure(text=f"{display_name}\n{activation:.2f}")
    
    def update_template_display(self):
        """Update the template activation display."""
        # Update features display
        self.features_text.delete(1.0, tk.END)
        
        if self.selected_syllable:
            self.features_text.insert(tk.END, f"TEMPLATE ACTIVATION FOR: {self.selected_syllable}\n")
            self.features_text.insert(tk.END, "=" * 50 + "\n\n")
        else:
            self.features_text.insert(tk.END, "TEMPLATE FEATURES OVERVIEW\n")
            self.features_text.insert(tk.END, "=" * 30 + "\n\n")
        
        # Show terminal activations
        self.features_text.insert(tk.END, "Terminal Features:\n")
        self.features_text.insert(tk.END, "-" * 20 + "\n")
        
        for terminal_id in self.terminals:
            if terminal_id in self.graph.units:
                unit = self.graph.units[terminal_id]
                state = unit.state.name
                activation = unit.a
                
                # Color code the text
                if activation > 0.7:
                    color = "green"
                elif activation > 0.4:
                    color = "orange"
                else:
                    color = "gray"
                
                self.features_text.insert(tk.END, f"{terminal_id}:\n")
                self.features_text.insert(tk.END, f"  State: {state}\n")
                self.features_text.insert(tk.END, f"  Activation: {activation:.3f}\n\n")
        
        # Show synthetic audio features
        self.features_text.insert(tk.END, "\nSynthetic Audio Features:\n")
        self.features_text.insert(tk.END, "-" * 25 + "\n")
        
        features = create_synthetic_audio_features("engage active perception")
        for feature_name, value in features.items():
            self.features_text.insert(tk.END, f"{feature_name}: {value:.3f}\n")
        
        # Update network state display
        self.state_text.delete(1.0, tk.END)
        
        self.state_text.insert(tk.END, "NETWORK STATE\n")
        self.state_text.insert(tk.END, "=" * 15 + "\n\n")
        
        # Show syllable states
        self.state_text.insert(tk.END, "Syllables:\n")
        self.state_text.insert(tk.END, "-" * 10 + "\n")
        
        for syllable in self.syllables:
            if syllable in self.graph.units:
                unit = self.graph.units[syllable]
                display_name = syllable.replace('u_', '').replace('_phoneme', '')
                self.state_text.insert(tk.END, f"{display_name}: {unit.state.name} ({unit.a:.2f})\n")
        
        # Show word states
        self.state_text.insert(tk.END, "\nWords:\n")
        self.state_text.insert(tk.END, "-" * 6 + "\n")
        
        for unit_id, unit in self.graph.units.items():
            if unit.kind == UnitType.SCRIPT and 'phoneme' not in unit_id and unit_id != 'u_phrase':
                display_name = unit_id.replace('u_', '')
                self.state_text.insert(tk.END, f"{display_name}: {unit.state.name} ({unit.a:.2f})\n")
        
        # Show phrase state
        if 'u_phrase' in self.graph.units:
            unit = self.graph.units['u_phrase']
            self.state_text.insert(tk.END, f"\nPhrase: {unit.state.name} ({unit.a:.2f})\n")


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = SyllableLearningApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application error: {e}")


if __name__ == "__main__":
    main()