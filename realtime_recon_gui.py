#!/usr/bin/env python3
"""
Real-time ReCoN GUI using tkinter

This module provides a real-time GUI for monitoring the "Engage Active Perception"
hypothesis using microphone input and visualizing the ReCoN network activation.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import queue
import sys
import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import networkx as nx
import pygame

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_capture import AudioCapture, AudioProcessor
from recon_core.compiler import compile_from_file
from recon_core.engine import Engine
from recon_core.config import EngineConfig
from recon_core.enums import UnitType, State

class ReCoNRealtimeGUI:
    """Real-time ReCoN visualization GUI."""
    
    def __init__(self):
        # Initialize pygame before creating tkinter widgets
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=1024)
        pygame.mixer.init()
        
        self.root = tk.Tk()
        self.root.title("ReCoN Real-time Audio Recognition: 'Engage Active Perception'")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.audio_capture = None
        self.audio_processor = AudioProcessor()
        self.recon_graph = None
        self.recon_engine = None
        
        # GUI state
        self.is_running = False
        self.update_queue = queue.Queue()
        
        # Create GUI elements
        self.create_widgets()
        
        # Start update loop
        self.update_gui()
        
        # Setup ReCoN network after GUI is created
        self.root.after(100, self.setup_recon_network)
        
    def create_widgets(self):
        """Create the GUI widgets."""
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        self.create_control_panel(main_frame)
        
        # Visualization area
        self.create_visualization_area(main_frame)
        
        # Status panel
        self.create_status_panel(main_frame)
    
    def create_control_panel(self, parent):
        """Create the control panel."""
        
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start/Stop buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(button_frame, text="Start Recognition")
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        self.start_button.config(command=self.start_recognition)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Recognition", state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        self.stop_button.config(command=self.stop_recognition)
        
        # Reset button
        self.reset_button = ttk.Button(button_frame, text="Reset Network")
        self.reset_button.pack(side=tk.LEFT, padx=(0, 10))
        self.reset_button.config(command=self.reset_network)
        
        # Settings frame
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Sensitivity slider
        ttk.Label(settings_frame, text="Sensitivity:").pack(side=tk.LEFT)
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                    variable=self.sensitivity_var, orient=tk.HORIZONTAL)
        sensitivity_scale.pack(side=tk.LEFT, padx=(5, 20), fill=tk.X, expand=True)
        
        # Speech detection threshold
        ttk.Label(settings_frame, text="Speech Threshold:").pack(side=tk.LEFT)
        self.speech_threshold_var = tk.DoubleVar(value=0.01)
        speech_scale = ttk.Scale(settings_frame, from_=0.001, to=0.1, 
                               variable=self.speech_threshold_var, orient=tk.HORIZONTAL)
        speech_scale.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
    
    def create_visualization_area(self, parent):
        """Create the visualization area."""
        
        viz_frame = ttk.Frame(parent)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Network graph tab
        self.create_network_tab()
        
        # Activation timeline tab
        self.create_timeline_tab()
        
        # Audio features tab
        self.create_audio_tab()
    
    def create_network_tab(self):
        """Create the network visualization tab."""
        
        network_frame = ttk.Frame(self.notebook)
        self.notebook.add(network_frame, text="ReCoN Network")
        
        # Create matplotlib figure
        self.network_fig = Figure(figsize=(12, 8), dpi=100)
        self.network_ax = self.network_fig.add_subplot(111)
        
        # Create canvas
        self.network_canvas = FigureCanvasTkAgg(self.network_fig, network_frame)
        self.network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty network
        self.draw_empty_network()
    
    def create_timeline_tab(self):
        """Create the activation timeline tab."""
        
        timeline_frame = ttk.Frame(self.notebook)
        self.notebook.add(timeline_frame, text="Activation Timeline")
        
        # Create matplotlib figure for timeline
        self.timeline_fig = Figure(figsize=(12, 6), dpi=100)
        self.timeline_ax = self.timeline_fig.add_subplot(111)
        
        # Create canvas
        self.timeline_canvas = FigureCanvasTkAgg(self.timeline_fig, timeline_frame)
        self.timeline_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize timeline
        self.timeline_data = {'time': [], 'units': {}}
        self.draw_empty_timeline()
    
    def create_audio_tab(self):
        """Create the audio features tab."""
        
        audio_frame = ttk.Frame(self.notebook)
        self.notebook.add(audio_frame, text="Audio Features")
        
        # Create matplotlib figure for audio features
        self.audio_fig = Figure(figsize=(12, 6), dpi=100)
        self.audio_ax = self.audio_fig.add_subplot(111)
        
        # Create canvas
        self.audio_canvas = FigureCanvasTkAgg(self.audio_fig, audio_frame)
        self.audio_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize audio features display
        self.audio_features = {}
        self.draw_empty_audio_features()
    
    def create_status_panel(self, parent):
        """Create the status panel."""
        
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Status labels
        self.status_label = ttk.Label(status_frame, text="Status: Ready")
        self.status_label.pack(side=tk.LEFT)
        
        self.recognition_label = ttk.Label(status_frame, text="Recognition: Inactive")
        self.recognition_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.audio_level_label = ttk.Label(status_frame, text="Audio Level: 0.0")
        self.audio_level_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          maximum=1.0, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=(20, 0))
    
    def setup_recon_network(self):
        """Setup the ReCoN network."""
        try:
            # Compile the graph
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'engage_active_perception.yaml')
            self.recon_graph = compile_from_file(script_path)
            
            # Create engine with configuration
            config = EngineConfig(
                sur_positive=0.4,
                por_positive=0.6,
                ret_positive=0.2,
                confirmation_ratio=0.75,
                deterministic_order=True,
                ret_feedback_enabled=True
            )
            
            self.recon_engine = Engine(self.recon_graph, config)
            
            # Draw initial network
            self.draw_network()
            
            print("ReCoN network initialized successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize ReCoN network: {e}")
            print(f"Error initializing ReCoN network: {e}")
    
    def start_recognition(self):
        """Start real-time recognition."""
        try:
            # Create audio capture
            self.audio_capture = AudioCapture(callback=self.audio_callback)
            
            if self.audio_capture.start_recording():
                self.is_running = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_label.config(text="Status: Recording")
                self.recognition_label.config(text="Recognition: Active")
                
                # Start processing thread
                self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
                self.processing_thread.start()
                
                print("Real-time recognition started")
            else:
                messagebox.showerror("Error", "Failed to start audio recording")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recognition: {e}")
            print(f"Error starting recognition: {e}")
    
    def stop_recognition(self):
        """Stop real-time recognition."""
        self.is_running = False
        
        if self.audio_capture:
            self.audio_capture.stop_recording()
            self.audio_capture = None
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        self.recognition_label.config(text="Recognition: Inactive")
        
        print("Real-time recognition stopped")
    
    def reset_network(self):
        """Reset the ReCoN network."""
        if self.recon_engine:
            self.recon_engine.reset()
            
            # Reset all units
            for unit in self.recon_graph.units.values():
                unit.state = State.INACTIVE
                unit.a = 0.0
                unit.inbox = []
                unit.outbox = []
            
            # Redraw network
            self.draw_network()
            self.draw_empty_timeline()
            self.draw_empty_audio_features()
            
            print("Network reset")
    
    def audio_callback(self, audio_data, timestamp):
        """Callback for audio data."""
        if self.is_running:
            # Put audio data in queue for processing
            self.update_queue.put(('audio', audio_data, timestamp))
    
    def process_audio(self):
        """Process audio data in a separate thread."""
        while self.is_running:
            try:
                # Get audio data
                audio = self.audio_capture.get_latest_audio(0.5)  # 0.5 second chunks
                
                if audio is not None:
                    # Extract features
                    features = self.audio_processor.extract_features(audio)
                    speech_detected = self.audio_processor.detect_speech_activity(
                        audio, self.speech_threshold_var.get())
                    
                    # Update audio features
                    self.audio_features = features
                    
                    # Calculate audio level
                    audio_level = np.sqrt(np.mean(audio**2))
                    
                    # Update ReCoN network if speech detected
                    if speech_detected and self.recon_engine:
                        self.update_recon_network(features)
                    
                    # Put update in queue
                    self.update_queue.put(('update', {
                        'features': features,
                        'speech_detected': speech_detected,
                        'audio_level': audio_level,
                        'timestamp': time.time()
                    }))
                
                time.sleep(0.1)  # Process every 100ms
                
            except Exception as e:
                print(f"Error in audio processing: {e}")
                time.sleep(0.1)
    
    def update_recon_network(self, features):
        """Update the ReCoN network with audio features."""
        try:
            # Map audio features to terminal units
            terminal_mapping = {
                't_mfcc_low': features.get('mfcc_low', 0.0),
                't_pitch_high': features.get('pitch_high', 0.0),
                't_rhythm': features.get('rhythm', 0.0),
                't_noise_level': features.get('noise_level', 0.0),
                't_formant': features.get('formant', 0.0),
                't_spectrogram': features.get('spectrogram', 0.0)
            }
            
            # Update terminal activations
            for terminal_id, activation in terminal_mapping.items():
                if terminal_id in self.recon_graph.units:
                    unit = self.recon_graph.units[terminal_id]
                    unit.a = activation
                    
                    # Set state based on activation
                    if activation > unit.thresh:
                        unit.state = State.TRUE
                    else:
                        unit.state = State.INACTIVE
            
            # Activate phrase if not already active
            if self.recon_graph.units['u_phrase'].state == State.INACTIVE:
                self.recon_graph.units['u_phrase'].a = 1.0
                self.recon_graph.units['u_phrase'].state = State.ACTIVE
            
            # Step the engine
            self.recon_engine.step(1)
            
        except Exception as e:
            print(f"Error updating ReCoN network: {e}")
    
    def draw_network(self):
        """Draw the ReCoN network."""
        if not self.recon_graph:
            return
        
        self.network_ax.clear()
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add units
        for unit_id, unit in self.recon_graph.units.items():
            G.add_node(unit_id, 
                      kind=unit.kind.name,
                      state=unit.state.name,
                      activation=unit.a,
                      threshold=unit.thresh)
        
        # Add edges
        for src_id, edges in self.recon_graph.out_edges.items():
            for edge in edges:
                G.add_edge(edge.src, edge.dst, 
                          type=edge.type.name, 
                          weight=edge.w)
        
        # Create layout
        pos = self.create_network_layout()
        
        # Draw edges by type
        edge_styles = {
            'SUB': {'color': '#2E8B57', 'style': 'solid', 'width': 2},
            'SUR': {'color': '#DC143C', 'style': 'solid', 'width': 1.5},
            'POR': {'color': '#8A2BE2', 'style': 'dashed', 'width': 2},
            'RET': {'color': '#FF8C00', 'style': 'dotted', 'width': 1.5}
        }
        
        for link_type, style in edge_styles.items():
            edges_of_type = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == link_type]
            if edges_of_type:
                nx.draw_networkx_edges(G, pos, edgelist=edges_of_type,
                                     edge_color=style['color'],
                                     style=style['style'],
                                     width=style['width'],
                                     alpha=0.7,
                                     arrows=True,
                                     arrowsize=15,
                                     ax=self.network_ax)
        
        # Draw nodes with state-based coloring
        node_colors = []
        node_sizes = []
        
        for unit_id in G.nodes():
            unit = self.recon_graph.units[unit_id]
            
            # Color based on state
            if unit.state == State.CONFIRMED:
                color = '#2CA25F'  # Green
                size = 800
            elif unit.state == State.ACTIVE:
                color = '#4A90E2'  # Blue
                size = 600
            elif unit.state == State.TRUE:
                color = '#31A354'  # Dark green
                size = 500
            elif unit.state == State.REQUESTED:
                color = '#6BAED6'  # Light blue
                size = 400
            else:
                color = '#CCCCCC'  # Gray
                size = 300
            
            node_colors.append(color)
            node_sizes.append(size)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, ax=self.network_ax)
        
        # Draw labels
        labels = {}
        for unit_id in G.nodes():
            unit = self.recon_graph.units[unit_id]
            if unit.kind == UnitType.SCRIPT:
                labels[unit_id] = f"{unit_id}\n{unit.state.name}\n{unit.a:.2f}"
            else:
                labels[unit_id] = f"{unit_id}\n{unit.a:.2f}"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=self.network_ax)
        
        self.network_ax.set_title('ReCoN Network: "Engage Active Perception"', fontsize=14, fontweight='bold')
        self.network_ax.axis('off')
        
        self.network_canvas.draw()
    
    def create_network_layout(self):
        """Create layout for the network."""
        pos = {}
        
        # Phrase level (top)
        pos['u_phrase'] = (0, 3)
        
        # Word level (second row)
        pos['u_engage'] = (-2, 2)
        pos['u_active'] = (0, 2)
        pos['u_perception'] = (2, 2)
        
        # Phoneme level (third row)
        pos['u_en_phoneme'] = (-2.5, 1)
        pos['u_gei_phoneme'] = (-2, 1)
        pos['u_dj_phoneme'] = (-1.5, 1)
        pos['u_ak_phoneme'] = (-0.5, 1)
        pos['u_ti_phoneme'] = (0, 1)
        pos['u_v_phoneme'] = (0.5, 1)
        pos['u_per_phoneme'] = (1.5, 1)
        pos['u_sep_phoneme'] = (2, 1)
        pos['u_shun_phoneme'] = (2.5, 1)
        
        # Terminal level (bottom)
        terminals = [uid for uid, u in self.recon_graph.units.items() if u.kind == UnitType.TERMINAL]
        for i, term_id in enumerate(terminals):
            x = -3 + (i * 6) / (len(terminals) - 1) if len(terminals) > 1 else 0
            pos[term_id] = (x, 0)
        
        return pos
    
    def draw_empty_network(self):
        """Draw empty network placeholder."""
        self.network_ax.text(0.5, 0.5, 'ReCoN Network\nClick "Start Recognition" to begin', 
                           ha='center', va='center', fontsize=16, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.network_ax.set_xlim(0, 1)
        self.network_ax.set_ylim(0, 1)
        self.network_ax.axis('off')
        self.network_canvas.draw()
    
    def draw_empty_timeline(self):
        """Draw empty timeline placeholder."""
        self.timeline_ax.text(0.5, 0.5, 'Activation Timeline\nWill show unit activations over time', 
                            ha='center', va='center', fontsize=16,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.timeline_ax.set_xlim(0, 1)
        self.timeline_ax.set_ylim(0, 1)
        self.timeline_ax.axis('off')
        self.timeline_canvas.draw()
    
    def draw_empty_audio_features(self):
        """Draw empty audio features placeholder."""
        self.audio_ax.text(0.5, 0.5, 'Audio Features\nWill show real-time audio feature extraction', 
                         ha='center', va='center', fontsize=16,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.audio_ax.set_xlim(0, 1)
        self.audio_ax.set_ylim(0, 1)
        self.audio_ax.axis('off')
        self.audio_canvas.draw()
    
    def update_gui(self):
        """Update the GUI (called periodically)."""
        try:
            # Check if root window still exists
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return
                
            # Process updates from queue
            while not self.update_queue.empty():
                try:
                    update_type, *data = self.update_queue.get_nowait()
                    
                    if update_type == 'update':
                        self.handle_update(data[0])
                    elif update_type == 'audio':
                        # Handle audio data if needed
                        pass
                        
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error processing update: {e}")
            
            # Update displays
            if self.is_running and self.recon_graph:
                self.draw_network()
                self.draw_audio_features()
            
            # Schedule next update
            self.root.after(100, self.update_gui)  # Update every 100ms
            
        except Exception as e:
            print(f"Error in GUI update: {e}")
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(100, self.update_gui)
    
    def handle_update(self, data):
        """Handle update data."""
        # Update audio level
        audio_level = data.get('audio_level', 0.0)
        self.audio_level_label.config(text=f"Audio Level: {audio_level:.3f}")
        
        # Update progress bar
        self.progress_var.set(min(audio_level * 10, 1.0))  # Scale audio level
        
        # Update recognition status
        if data.get('speech_detected', False):
            self.recognition_label.config(text="Recognition: Speech Detected")
        else:
            self.recognition_label.config(text="Recognition: Listening")
    
    def draw_audio_features(self):
        """Draw audio features."""
        if not self.audio_features:
            return
        
        self.audio_ax.clear()
        
        # Create bar chart of features
        features = list(self.audio_features.keys())
        values = list(self.audio_features.values())
        
        bars = self.audio_ax.bar(features, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', 
                                                         '#96CEB4', '#FECA57', '#FF9FF3'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.audio_ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{value:.3f}', ha='center', va='bottom')
        
        self.audio_ax.set_ylim(0, 1.0)
        self.audio_ax.set_ylabel('Feature Value')
        self.audio_ax.set_title('Real-time Audio Features')
        self.audio_ax.tick_params(axis='x', rotation=45)
        
        self.audio_canvas.draw()
    
    def run(self):
        """Run the GUI."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("GUI interrupted by user")
        finally:
            self.stop_recognition()
            pygame.mixer.quit()

def main():
    """Main function."""
    print("Starting ReCoN Real-time Audio Recognition GUI...")
    
    try:
        app = ReCoNRealtimeGUI()
        app.run()
    except Exception as e:
        print(f"Error running GUI: {e}")
        messagebox.showerror("Error", f"Failed to start GUI: {e}")

if __name__ == "__main__":
    main()