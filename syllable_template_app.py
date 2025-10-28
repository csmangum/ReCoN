#!/usr/bin/env python3
"""
Syllable Template Interactive Learning App

A tkinter application that visualizes how syllable/phoneme templates are activated
in the ReCoN (Request Confirmation Network) system. This helps users understand
how terminal scripts work and how to make them more accurate.

Features:
- Interactive buttons for each phoneme/syllable
- Real-time visualization of template activation
- Audio feature terminal display
- Network state visualization
- Step-by-step simulation controls
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import threading
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


class SyllableTemplateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Syllable Template Interactive Learning App")
        self.root.geometry("1200x800")
        
        # Load the audio phrase recognition network
        self.load_network()
        
        # Initialize UI
        self.setup_ui()
        
        # Current simulation state
        self.current_step = 0
        self.max_steps = 20
        self.is_running = False
        
    def load_network(self):
        """Load the audio phrase recognition network from YAML."""
        try:
            import yaml
            with open('/workspace/scripts/engage_active_perception.yaml', 'r') as f:
                network_spec = yaml.safe_load(f)
            
            # Compile the network
            self.graph = compile_from_dict(network_spec)
            
            # Create engine with configuration
            config = EngineConfig()
            config.deterministic_order = True
            config.confirmation_ratio = 0.75
            config.sur_positive = 0.4
            config.ret_feedback_enabled = True
            
            self.engine = Engine(self.graph, config)
            
            # Extract phonemes and terminals
            self.phonemes = []
            self.terminals = []
            self.words = []
            
            for unit_id, unit in self.graph.units.items():
                if unit.kind == UnitType.SCRIPT:
                    if 'phoneme' in unit_id:
                        self.phonemes.append(unit_id)
                    elif unit_id != 'u_phrase':
                        self.words.append(unit_id)
                elif unit.kind == UnitType.TERMINAL:
                    self.terminals.append(unit_id)
            
            print(f"Loaded network with {len(self.phonemes)} phonemes, {len(self.words)} words, {len(self.terminals)} terminals")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load network: {e}")
            self.root.quit()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Syllable Template Interactive Learning", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Create notebook for different views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Phoneme buttons tab
        self.setup_phoneme_tab(notebook)
        
        # Network visualization tab
        self.setup_network_tab(notebook)
        
        # Terminal features tab
        self.setup_terminal_tab(notebook)
        
        # Simulation log tab
        self.setup_log_tab(notebook)
    
    def setup_control_panel(self, parent):
        """Set up the control panel with simulation controls."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Step controls
        ttk.Label(control_frame, text="Simulation Controls:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.step_var = tk.StringVar(value="0")
        step_entry = ttk.Entry(control_frame, textvariable=self.step_var, width=5)
        step_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Step", command=self.step_simulation).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Run", command=self.run_simulation).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Stop", command=self.stop_simulation).pack(side=tk.LEFT, padx=(0, 5))
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.RIGHT)
    
    def setup_phoneme_tab(self, notebook):
        """Set up the phoneme buttons tab."""
        phoneme_frame = ttk.Frame(notebook)
        notebook.add(phoneme_frame, text="Phoneme Templates")
        
        # Instructions
        instructions = ttk.Label(phoneme_frame, 
                               text="Click on phoneme buttons to see how they activate template features.\n" +
                                    "Green = CONFIRMED, Yellow = ACTIVE, Orange = REQUESTED, Gray = INACTIVE",
                               font=('Arial', 10))
        instructions.pack(pady=10)
        
        # Phoneme buttons frame
        phoneme_buttons_frame = ttk.Frame(phoneme_frame)
        phoneme_buttons_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create phoneme buttons in a grid
        self.phoneme_buttons = {}
        cols = 4
        for i, phoneme in enumerate(self.phonemes):
            row = i // cols
            col = i % cols
            
            # Extract phoneme name for display
            display_name = phoneme.replace('u_', '').replace('_phoneme', '')
            
            btn = tk.Button(phoneme_buttons_frame, 
                          text=display_name,
                          command=lambda p=phoneme: self.activate_phoneme(p),
                          width=12, height=2,
                          font=('Arial', 10))
            btn.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            
            self.phoneme_buttons[phoneme] = btn
        
        # Configure grid weights
        for i in range(cols):
            phoneme_buttons_frame.columnconfigure(i, weight=1)
    
    def setup_network_tab(self, notebook):
        """Set up the network visualization tab."""
        network_frame = ttk.Frame(notebook)
        notebook.add(network_frame, text="Network State")
        
        # Create canvas for network visualization
        self.network_canvas = tk.Canvas(network_frame, bg='white', width=800, height=600)
        self.network_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Draw the network
        self.draw_network()
    
    def setup_terminal_tab(self, notebook):
        """Set up the terminal features tab."""
        terminal_frame = ttk.Frame(notebook)
        notebook.add(terminal_frame, text="Terminal Features")
        
        # Terminal features display
        self.terminal_text = tk.Text(terminal_frame, height=20, width=80, font=('Courier', 10))
        terminal_scrollbar = ttk.Scrollbar(terminal_frame, orient=tk.VERTICAL, command=self.terminal_text.yview)
        self.terminal_text.configure(yscrollcommand=terminal_scrollbar.set)
        
        self.terminal_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        terminal_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Update terminal display
        self.update_terminal_display()
    
    def setup_log_tab(self, notebook):
        """Set up the simulation log tab."""
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Simulation Log")
        
        # Log text widget
        self.log_text = tk.Text(log_frame, height=20, width=80, font=('Courier', 9))
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    
    def activate_phoneme(self, phoneme_id):
        """Activate a specific phoneme and show its effects."""
        self.log_message(f"Activating phoneme: {phoneme_id}")
        
        # Reset the network
        self.engine.reset()
        
        # Manually activate the phoneme
        if phoneme_id in self.graph.units:
            unit = self.graph.units[phoneme_id]
            unit.state = State.ACTIVE
            unit.a = 1.0
            
            # Run a few steps to see the effects
            for _ in range(5):
                self.engine.step()
                self.update_displays()
                time.sleep(0.1)
        
        self.log_message(f"Phoneme {phoneme_id} activation complete")
    
    def reset_simulation(self):
        """Reset the simulation to initial state."""
        self.engine.reset()
        self.current_step = 0
        self.step_var.set("0")
        self.status_var.set("Reset")
        self.update_displays()
        self.log_message("Simulation reset")
    
    def step_simulation(self):
        """Run one step of the simulation."""
        if self.current_step < self.max_steps:
            self.engine.step()
            self.current_step += 1
            self.step_var.set(str(self.current_step))
            self.status_var.set(f"Step {self.current_step}")
            self.update_displays()
            self.log_message(f"Step {self.current_step} completed")
        else:
            self.status_var.set("Max steps reached")
    
    def run_simulation(self):
        """Run the full simulation."""
        if self.is_running:
            return
        
        self.is_running = True
        self.status_var.set("Running...")
        
        def run_loop():
            while self.is_running and self.current_step < self.max_steps:
                self.engine.step()
                self.current_step += 1
                self.step_var.set(str(self.current_step))
                
                # Update UI in main thread
                self.root.after(0, self.update_displays)
                self.root.after(0, lambda: self.log_message(f"Step {self.current_step} completed"))
                
                time.sleep(0.5)  # Slow down for visualization
            
            self.is_running = False
            self.root.after(0, lambda: self.status_var.set("Completed"))
        
        thread = threading.Thread(target=run_loop)
        thread.daemon = True
        thread.start()
    
    def stop_simulation(self):
        """Stop the running simulation."""
        self.is_running = False
        self.status_var.set("Stopped")
    
    def update_displays(self):
        """Update all display elements."""
        self.update_phoneme_buttons()
        self.update_terminal_display()
        self.draw_network()
    
    def update_phoneme_buttons(self):
        """Update phoneme button colors based on their states."""
        for phoneme_id, button in self.phoneme_buttons.items():
            if phoneme_id in self.graph.units:
                unit = self.graph.units[phoneme_id]
                state = unit.state
                activation = unit.a
                
                # Color coding based on state
                if state == State.CONFIRMED:
                    color = 'lightgreen'
                elif state == State.ACTIVE:
                    color = 'yellow'
                elif state == State.REQUESTED:
                    color = 'orange'
                else:
                    color = 'lightgray'
                
                button.configure(bg=color)
                
                # Update button text with activation level
                display_name = phoneme_id.replace('u_', '').replace('_phoneme', '')
                button.configure(text=f"{display_name}\n{activation:.2f}")
    
    def update_terminal_display(self):
        """Update the terminal features display."""
        self.terminal_text.delete(1.0, tk.END)
        
        # Get current terminal activations
        terminal_activations = {}
        for terminal_id in self.terminals:
            if terminal_id in self.graph.units:
                unit = self.graph.units[terminal_id]
                terminal_activations[terminal_id] = {
                    'state': unit.state.name,
                    'activation': unit.a
                }
        
        # Display terminal information
        self.terminal_text.insert(tk.END, "Terminal Features Status:\n")
        self.terminal_text.insert(tk.END, "=" * 50 + "\n\n")
        
        for terminal_id, info in terminal_activations.items():
            self.terminal_text.insert(tk.END, f"{terminal_id}:\n")
            self.terminal_text.insert(tk.END, f"  State: {info['state']}\n")
            self.terminal_text.insert(tk.END, f"  Activation: {info['activation']:.3f}\n\n")
        
        # Show synthetic audio features
        self.terminal_text.insert(tk.END, "\nSynthetic Audio Features:\n")
        self.terminal_text.insert(tk.END, "=" * 30 + "\n")
        
        features = create_synthetic_audio_features("engage active perception")
        for feature_name, value in features.items():
            self.terminal_text.insert(tk.END, f"{feature_name}: {value:.3f}\n")
    
    def draw_network(self):
        """Draw the network visualization."""
        self.network_canvas.delete("all")
        
        # Simple network layout
        width = self.network_canvas.winfo_width()
        height = self.network_canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # Draw units as circles
        unit_positions = {}
        
        # Position units in a hierarchical layout
        y_offset = 50
        x_spacing = width // (len(self.phonemes) + 1)
        
        # Draw phonemes
        for i, phoneme in enumerate(self.phonemes):
            x = (i + 1) * x_spacing
            y = height - 100
            unit_positions[phoneme] = (x, y)
            
            # Get unit state and activation
            if phoneme in self.graph.units:
                unit = self.graph.units[phoneme]
                state = unit.state
                activation = unit.a
                
                # Color based on state
                if state == State.CONFIRMED:
                    color = 'green'
                elif state == State.ACTIVE:
                    color = 'yellow'
                elif state == State.REQUESTED:
                    color = 'orange'
                else:
                    color = 'gray'
                
                # Draw circle
                radius = 20 + int(activation * 20)  # Size based on activation
                self.network_canvas.create_oval(x - radius, y - radius, 
                                              x + radius, y + radius,
                                              fill=color, outline='black', width=2)
                
                # Draw label
                display_name = phoneme.replace('u_', '').replace('_phoneme', '')
                self.network_canvas.create_text(x, y, text=display_name, font=('Arial', 8))
        
        # Draw words
        word_y = height - 200
        word_x_spacing = width // (len(self.words) + 1)
        
        for i, word in enumerate(self.words):
            x = (i + 1) * word_x_spacing
            y = word_y
            unit_positions[word] = (x, y)
            
            if word in self.graph.units:
                unit = self.graph.units[word]
                state = unit.state
                activation = unit.a
                
                if state == State.CONFIRMED:
                    color = 'green'
                elif state == State.ACTIVE:
                    color = 'yellow'
                elif state == State.REQUESTED:
                    color = 'orange'
                else:
                    color = 'gray'
                
                radius = 25 + int(activation * 25)
                self.network_canvas.create_oval(x - radius, y - radius,
                                              x + radius, y + radius,
                                              fill=color, outline='black', width=2)
                
                display_name = word.replace('u_', '')
                self.network_canvas.create_text(x, y, text=display_name, font=('Arial', 10, 'bold'))
        
        # Draw phrase
        phrase_x = width // 2
        phrase_y = 50
        unit_positions['u_phrase'] = (phrase_x, phrase_y)
        
        if 'u_phrase' in self.graph.units:
            unit = self.graph.units['u_phrase']
            state = unit.state
            activation = unit.a
            
            if state == State.CONFIRMED:
                color = 'green'
            elif state == State.ACTIVE:
                color = 'yellow'
            elif state == State.REQUESTED:
                color = 'orange'
            else:
                color = 'gray'
            
            radius = 30 + int(activation * 30)
            self.network_canvas.create_oval(phrase_x - radius, phrase_y - radius,
                                          phrase_x + radius, phrase_y + radius,
                                          fill=color, outline='black', width=3)
            
            self.network_canvas.create_text(phrase_x, phrase_y, text="PHRASE", 
                                          font=('Arial', 12, 'bold'))
    
    def log_message(self, message):
        """Add a message to the log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = SyllableTemplateApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application error: {e}")


if __name__ == "__main__":
    main()