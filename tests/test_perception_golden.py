"""
Tiny golden baseline tests for perception features.

These are intentionally lightweight and robust across minor numerical changes.
"""

from perception.dataset import make_house_scene, make_barn_scene
from perception.terminals import terminals_from_image, advanced_terminals_from_image


def test_house_basic_terminals_have_reasonable_values():
	img = make_house_scene(size=64, noise=0.0)
	feats = terminals_from_image(img)

	# Mean intensity should be within a broad, sane range
	assert 0.05 <= feats['t_mean'] <= 0.95
	# Edge responses should be positive
	assert feats['t_vert'] > 0.0
	assert feats['t_horz'] > 0.0


def test_barn_is_wider_than_house_by_geometric_aspect_proxy():
	# Barn is designed to be wider; our geometric feature t_aspect reflects width/height ratio
	house = make_house_scene(size=64, noise=0.0, scale_factor=1.0)
	barn = make_barn_scene(size=64, noise=0.0, scale_factor=1.0)

	house_feats = advanced_terminals_from_image(house)
	barn_feats = advanced_terminals_from_image(barn)

	# Barn aspect proxy should be at least as large as house (allowing equality in corner cases)
	assert barn_feats['t_aspect'] >= house_feats['t_aspect']
	# Both should be within expected normalized range [0, 0.3]
	assert 0.0 <= house_feats['t_aspect'] <= 0.3
	assert 0.0 <= barn_feats['t_aspect'] <= 0.3

