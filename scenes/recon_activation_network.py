from manim import *

# High-level scene introducing a hypothetical Recon Activation Network and its core components
class ReconActivationNetwork(Scene):
	def construct(self):
		title = Text("Recon Activation Network", weight=BOLD).scale(0.9)
		subtitle = Text("Architecture Overview", slant=ITALIC).scale(0.5).next_to(title, DOWN, buff=0.2)
		title_group = VGroup(title, subtitle).to_edge(UP)
		self.play(FadeIn(title_group, shift=UP), run_time=1.0)
		self.wait(0.2)

		# Define component cards
		def component_card(heading: str, bullets: list[str], color: ManimColor = BLUE_E):
			header = Text(heading, weight=BOLD).scale(0.45)
			items = VGroup(*[Text(f"- {b}").scale(0.35).set_opacity(0.9) for b in bullets])
			items.arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(header, DOWN, aligned_edge=LEFT)
			card = VGroup(header, items)
			card.set_backstroke(width=4, color=BLACK)
			card.set_stroke(color=color, width=2)
			card.set_fill(color=color, opacity=0.10)
			card.background_rectangle = SurroundingRectangle(card, corner_radius=0.12, stroke_width=2, color=color)
			container = VGroup(card.background_rectangle, card).set_z_index(1)
			return container

		# Components definitions
		input_card = component_card(
			"Input Stream",
			[
				"Sensor frames / events",
				"Timestamps, meta",
				"Batch N × T × C × H × W",
			],
			color=TEAL_E,
		)

		preprocess_card = component_card(
			"Preprocessing",
			[
				"Denoise, normalize",
				"Patchify / tokenization",
				"Positional encodings",
			],
			color=BLUE_E,
		)

		feature_encoder_card = component_card(
			"Feature Encoder",
			[
				"CNN/ViT hybrid",
				"Multi-scale features",
				"Latent Z ∈ ℝ^{N×L×D}",
			],
			color=PURPLE_E,
		)

		reconstruction_head_card = component_card(
			"Reconstruction Head",
			[
				"Decoder (Conv/UpSample)",
				"Reconstruct input modality",
				"MSE/SSIM losses",
			],
			color=MAROON_E,
		)

		activation_head_card = component_card(
			"Activation Head",
			[
				"Event triggers / logits",
				"Temporal aggregation",
				"Calibrated thresholds",
			],
			color=GREEN_E,
		)

		monitoring_card = component_card(
			"Monitoring",
			[
				"Latency, drift, health",
				"Alerts & dashboards",
				"A/B and canaries",
			],
			color=ORANGE,
		)

		# Layout
		row1 = VGroup(input_card, preprocess_card, feature_encoder_card).arrange(RIGHT, buff=0.6, aligned_edge=UP).scale(0.9)
		row2 = VGroup(reconstruction_head_card, activation_head_card, monitoring_card).arrange(RIGHT, buff=0.6, aligned_edge=UP).scale(0.9)
		row2.next_to(row1, DOWN, buff=0.6)
		grid = VGroup(row1, row2).move_to(ORIGIN).shift(DOWN * 0.3)

		all_cards = [
			input_card,
			preprocess_card,
			feature_encoder_card,
			reconstruction_head_card,
			activation_head_card,
			monitoring_card,
		]
		self.play(LaggedStart(*[FadeIn(card[0], shift=UP*0.2) for card in all_cards], lag_ratio=0.05), run_time=1.2)
		self.play(LaggedStart(*[FadeIn(card[1], shift=UP*0.1) for card in all_cards], lag_ratio=0.05), run_time=1.2)
		self.wait(0.2)

		# Connectors (flow arrows)
		def arrow_between(a: Mobject, b: Mobject, color: ManimColor = GREY_B):
			return Arrow(a.get_right(), b.get_left(), buff=0.2, stroke_width=3, max_tip_length_to_length_ratio=0.08, color=color)

		arrows = VGroup(
			arrow_between(input_card, preprocess_card),
			arrow_between(preprocess_card, feature_encoder_card),
			arrow_between(feature_encoder_card, reconstruction_head_card, color=MAROON_E),
			arrow_between(feature_encoder_card, activation_head_card, color=GREEN_E),
			arrow_between(activation_head_card, monitoring_card, color=ORANGE),
		)
		self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.2), run_time=1.2)
		self.wait(0.2)

		# Animated data packet
		packet = Dot(color=YELLOW).scale(0.7)

		def move_packet_along(arrow: Arrow, color: ManimColor, run_time: float = 1.0):
			return MoveAlongPath(packet.set_color(color), arrow, rate_func=linear, run_time=run_time)

		self.play(FadeIn(packet))
		self.play(move_packet_along(arrows[0], TEAL_E, 0.8))
		self.play(Indicate(preprocess_card[1], color=BLUE_E, scale_factor=1.02))
		self.play(move_packet_along(arrows[1], BLUE_E, 0.9))
		self.play(Wiggle(feature_encoder_card[1], rotation_angle=0.02))
		self.play(
			AnimationGroup(
				move_packet_along(arrows[2], MAROON_E, 0.9),
				move_packet_along(arrows[3], GREEN_E, 0.9),
				lag_ratio=0.15,
			)
		)
		self.play(Flash(activation_head_card[0], color=GREEN_E, flash_radius=0.6))
		self.wait(0.2)

		# Spotlight explanation panel
		panel = RoundedRectangle(corner_radius=0.15, width=6.4, height=1.6, stroke_width=2, color=GREY_B)
		panel.set_fill(GREY_E, opacity=0.15)
		panel_text = Text(
			"Recon Activation Network: encodes inputs, reconstructs for self-supervision,\n"
			"and emits activations for events with calibrated thresholds.",
			t2c={"encodes": BLUE_E, "reconstructs": MAROON_E, "activations": GREEN_E, "calibrated": YELLOW},
		).scale(0.38)
		panel_group = VGroup(panel, panel_text).arrange(LEFT, buff=0.4).to_edge(DOWN).shift(UP*0.1)
		self.play(FadeIn(panel_group, shift=UP*0.2))

		# Legends
		legend_ar = VGroup(
			VGroup(Line(ORIGIN, RIGHT*0.8).set_stroke(MAROON_E, 3), Text("Reconstruction path").scale(0.32)).arrange(RIGHT, buff=0.2),
			VGroup(Line(ORIGIN, RIGHT*0.8).set_stroke(GREEN_E, 3), Text("Activation path").scale(0.32)).arrange(RIGHT, buff=0.2),
		).arrange(RIGHT, buff=0.6).scale(0.9)
		legend_ar.next_to(panel_group, UP, buff=0.2)
		self.play(FadeIn(legend_ar, shift=UP*0.1))

		# Focus each component briefly
		for comp, col in [
			(input_card, TEAL_E),
			(preprocess_card, BLUE_E),
			(feature_encoder_card, PURPLE_E),
			(reconstruction_head_card, MAROON_E),
			(activation_head_card, GREEN_E),
			(monitoring_card, ORANGE),
		]:
			self.play(Circumscribe(comp, color=col, buff=0.12), run_time=0.6)
			self.wait(0.05)

		# Closing
		thanks = Text("Data in. Insights out.", slant=ITALIC).scale(0.5).next_to(panel_group, UP, buff=0.4)
		self.play(Write(thanks))
		self.wait(0.6)
		self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)