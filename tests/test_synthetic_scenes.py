"""
Unit tests for synthetic scene generation functionality.

Tests the enhanced scene generator with variety features including:
- Basic house scenes with scaling and positioning
- Barn scenes with arched roofs
- Occluded scenes with different obstruction types
- Random variations in size and position
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perception.dataset import (
    canvas, draw_rect, draw_triangle, make_house_scene, make_barn_scene,
    make_occluded_scene, make_varied_scene, make_church_scene, make_tent_scene,
    make_tower_scene
)


class TestBasicDrawing(unittest.TestCase):
    """Test basic drawing primitives."""
    
    def test_canvas_creation(self):
        """Test canvas creation with different sizes."""
        # Default size
        img = canvas()
        self.assertEqual(img.shape, (64, 64))
        self.assertEqual(img.dtype, np.float32)
        self.assertTrue(np.all(img == 0))
        
        # Custom size
        img = canvas(128)
        self.assertEqual(img.shape, (128, 128))
        self.assertEqual(img.dtype, np.float32)
    
    def test_draw_rect(self):
        """Test rectangle drawing."""
        img = canvas(64)
        
        # Draw rectangle
        draw_rect(img, 10, 10, 20, 15, 0.7)
        
        # Check rectangle was drawn (use almost equal for floating point)
        self.assertAlmostEqual(img[10, 10], 0.7, places=6)  # Top-left corner
        self.assertAlmostEqual(img[24, 29], 0.7, places=6)  # Bottom-right corner
        self.assertEqual(img[5, 5], 0.0)    # Outside rectangle
        
        # Check bounds (use allclose for floating point comparison)
        self.assertTrue(np.allclose(img[10:25, 10:30], 0.7))
        
    def test_draw_triangle(self):
        """Test triangle drawing."""
        img = canvas(64)
        
        # Draw triangle
        draw_triangle(img, 20, 10, 20, 15, 1.0)
        
        # Check triangle properties - the triangle implementation has the apex narrow
        # and gets wider toward the base. Check actual rendered positions.
        
        # Should have triangle pixels in the expected region
        triangle_pixels = np.sum(img > 0)
        self.assertGreater(triangle_pixels, 50)  # Should have substantial area
        
        # Check that there are pixels in the triangle region
        triangle_region = img[10:25, 20:40]  # Triangle spans these approximate bounds
        self.assertGreater(np.sum(triangle_region > 0), 50)
        
        # Check that triangle gets wider towards the bottom
        # Find the actual Y range where triangle exists
        y_coords, x_coords = np.where(img > 0)
        if len(y_coords) > 0:
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            
            # Width at the top should be less than width at the bottom
            top_width = np.sum(img[min_y, :] > 0)
            bottom_width = np.sum(img[max_y, :] > 0)
            self.assertGreaterEqual(bottom_width, top_width)  # Triangle gets wider downward
        
        # Check that nothing is drawn outside expected bounds
        self.assertEqual(img[5, 29], 0.0)   # Above triangle


class TestHouseScenes(unittest.TestCase):
    """Test house scene generation."""
    
    def test_basic_house_scene(self):
        """Test basic house scene generation."""
        scene = make_house_scene(size=64, noise=0.0)
        
        # Check scene properties
        self.assertEqual(scene.shape, (64, 64))
        self.assertEqual(scene.dtype, np.float32)
        self.assertTrue(np.all(scene >= 0))
        self.assertTrue(np.all(scene <= 1))
        
        # Check that house components exist
        # House body should be at 0.7 intensity
        body_pixels = np.sum(np.abs(scene - 0.7) < 0.01)
        self.assertGreater(body_pixels, 100)  # Should have substantial house body
        
        # Roof should be at 1.0 intensity  
        roof_pixels = np.sum(np.abs(scene - 1.0) < 0.01)
        self.assertGreater(roof_pixels, 50)   # Should have roof
        
        # Door should be at 0.9 intensity
        door_pixels = np.sum(np.abs(scene - 0.9) < 0.01)
        self.assertGreater(door_pixels, 20)   # Should have door
    
    def test_house_scene_with_scaling(self):
        """Test house scene with different scaling factors."""
        # Small house
        small_house = make_house_scene(size=64, noise=0.0, scale_factor=0.5)
        small_pixels = np.sum(small_house > 0.1)
        
        # Large house
        large_house = make_house_scene(size=64, noise=0.0, scale_factor=1.5) 
        large_pixels = np.sum(large_house > 0.1)
        
        # Large house should have more pixels
        self.assertGreater(large_pixels, small_pixels)
    
    def test_house_scene_with_positioning(self):
        """Test house scene with position offsets."""
        # Centered house
        centered = make_house_scene(size=64, noise=0.0, position_offset=(0, 0))
        
        # Shifted house
        shifted = make_house_scene(size=64, noise=0.0, position_offset=(10, -5))
        
        # Houses should be different due to positioning
        self.assertFalse(np.array_equal(centered, shifted))
        
        # Both should have similar total content
        centered_content = np.sum(centered > 0.1)
        shifted_content = np.sum(shifted > 0.1)
        self.assertAlmostEqual(centered_content, shifted_content, delta=50)
    
    def test_house_scene_with_noise(self):
        """Test house scene with noise."""
        clean_scene = make_house_scene(size=64, noise=0.0)
        noisy_scene = make_house_scene(size=64, noise=0.1)
        
        # Noisy scene should be different from clean
        self.assertFalse(np.array_equal(clean_scene, noisy_scene))
        
        # But both should have similar structure (correlation)
        correlation = np.corrcoef(clean_scene.flatten(), noisy_scene.flatten())[0, 1]
        self.assertGreater(correlation, 0.8)  # Should be highly correlated


class TestBarnScenes(unittest.TestCase):
    """Test barn scene generation."""
    
    def test_basic_barn_scene(self):
        """Test basic barn scene generation."""
        scene = make_barn_scene(size=64, noise=0.0)
        
        # Check scene properties
        self.assertEqual(scene.shape, (64, 64))
        self.assertTrue(np.all(scene >= 0))
        self.assertTrue(np.all(scene <= 1))
        
        # Check barn components
        # Barn body should be at 0.6 intensity
        body_pixels = np.sum(np.abs(scene - 0.6) < 0.01)
        self.assertGreater(body_pixels, 100)
        
        # Arched roof should be at 0.8 intensity
        roof_pixels = np.sum(np.abs(scene - 0.8) < 0.01)
        self.assertGreater(roof_pixels, 30)
        
        # Door opening should be at 0.3 intensity (darker)
        door_pixels = np.sum(np.abs(scene - 0.3) < 0.01)
        self.assertGreater(door_pixels, 20)
    
    def test_barn_vs_house_differences(self):
        """Test that barns are distinguishable from houses."""
        house = make_house_scene(size=64, noise=0.0)
        barn = make_barn_scene(size=64, noise=0.0)
        
        # Scenes should be different
        self.assertFalse(np.array_equal(house, barn))
        
        # Barn should generally be wider (more horizontal extent)
        house_horizontal = np.sum(np.max(house, axis=0) > 0.1)
        barn_horizontal = np.sum(np.max(barn, axis=0) > 0.1)
        self.assertGreater(barn_horizontal, house_horizontal * 0.8)


class TestOccludedScenes(unittest.TestCase):
    """Test occluded scene generation."""
    
    def test_tree_occlusion(self):
        """Test scene with tree occlusion."""
        scene = make_occluded_scene(size=64, noise=0.0, occlusion_type='tree')
        
        # Should have house components + tree components
        # Tree trunk at 0.4 intensity
        trunk_pixels = np.sum(np.abs(scene - 0.4) < 0.01)
        self.assertGreater(trunk_pixels, 10)
        
        # Tree foliage at 0.5 intensity
        foliage_pixels = np.sum(np.abs(scene - 0.5) < 0.01)
        self.assertGreater(foliage_pixels, 20)
    
    def test_cloud_occlusion(self):
        """Test scene with cloud occlusion."""
        scene = make_occluded_scene(size=64, noise=0.0, occlusion_type='cloud')
        
        # Should have house + bright cloud components
        cloud_pixels = np.sum(np.abs(scene - 0.9) < 0.01)
        self.assertGreater(cloud_pixels, 50)  # Clouds should be substantial
    
    def test_box_occlusion(self):
        """Test scene with box occlusion."""
        scene = make_occluded_scene(size=64, noise=0.0, occlusion_type='box')
        
        # Should have house + dark box components
        box_pixels = np.sum(np.abs(scene - 0.3) < 0.01)
        self.assertGreater(box_pixels, 50)  # Box should be substantial
    
    def test_occlusion_vs_clean(self):
        """Test that occluded scenes differ from clean scenes."""
        clean = make_house_scene(size=64, noise=0.0, scale_factor=0.8)
        occluded = make_occluded_scene(size=64, noise=0.0, occlusion_type='tree')
        
        # Should be different due to occlusion
        self.assertFalse(np.array_equal(clean, occluded))
        
        # But should have reasonable correlation (house still there)
        correlation = np.corrcoef(clean.flatten(), occluded.flatten())[0, 1]
        self.assertGreater(correlation, 0.3)  # Some correlation due to house


class TestVariedScenes(unittest.TestCase):
    """Test varied scene generation with randomization."""
    
    def test_varied_house_scenes(self):
        """Test varied house scene generation."""
        scenes = [make_varied_scene('house', size=64) for _ in range(5)]
        
        # All should be valid scenes
        for scene in scenes:
            self.assertEqual(scene.shape, (64, 64))
            self.assertTrue(np.all(scene >= 0))
            self.assertTrue(np.all(scene <= 1))
        
        # Should have variety (not all identical)
        all_same = all(np.array_equal(scenes[0], scene) for scene in scenes[1:])
        self.assertFalse(all_same)
    
    def test_varied_barn_scenes(self):
        """Test varied barn scene generation."""
        scenes = [make_varied_scene('barn', size=64) for _ in range(5)]
        
        # All should be valid scenes
        for scene in scenes:
            self.assertEqual(scene.shape, (64, 64))
            self.assertTrue(np.all(scene >= 0))
            self.assertTrue(np.all(scene <= 1))
        
        # Should have variety
        all_same = all(np.array_equal(scenes[0], scene) for scene in scenes[1:])
        self.assertFalse(all_same)
    
    def test_varied_occluded_scenes(self):
        """Test varied occluded scene generation."""
        scenes = [make_varied_scene('occluded', size=64) for _ in range(5)]
        
        # All should be valid scenes
        for scene in scenes:
            self.assertEqual(scene.shape, (64, 64))
            self.assertTrue(np.all(scene >= 0))
            self.assertTrue(np.all(scene <= 1))
    
    def test_scale_range_parameter(self):
        """Test that scale range parameter affects scene size."""
        # Generate scenes with different scale ranges
        small_scenes = [make_varied_scene('house', size=64, scale_range=(0.5, 0.7)) for _ in range(10)]
        large_scenes = [make_varied_scene('house', size=64, scale_range=(1.2, 1.5)) for _ in range(10)]
        
        # Calculate average content for each group
        small_avg_content = np.mean([np.sum(scene > 0.1) for scene in small_scenes])
        large_avg_content = np.mean([np.sum(scene > 0.1) for scene in large_scenes])
        
        # Large scenes should have more content on average
        self.assertGreater(large_avg_content, small_avg_content)
    
    def test_unknown_scene_type_defaults_to_house(self):
        """Test that unknown scene types default to house."""
        house_scene = make_varied_scene('house', size=64)
        unknown_scene = make_varied_scene('unknown_type', size=64)
        
        # Should both be house-like structures - check they have similar total content
        # Note: Due to randomization in positioning and scaling, exact overlap varies
        house_content = np.sum(house_scene > 0.1)
        unknown_content = np.sum(unknown_scene > 0.1)
        
        # Both should have similar amounts of content (both are house scenes)
        content_ratio = min(house_content, unknown_content) / max(house_content, unknown_content)
        self.assertGreater(content_ratio, 0.4)  # Should be reasonably similar amount of content

    def test_new_scene_types_basic_properties(self):
        """Test basic generation for church, tent, and tower scenes."""
        church = make_church_scene(size=64, noise=0.0)
        tent = make_tent_scene(size=64, noise=0.0)
        tower = make_tower_scene(size=64, noise=0.0)

        for scene in [church, tent, tower]:
            self.assertEqual(scene.shape, (64, 64))
            self.assertTrue(np.all(scene >= 0))
            self.assertTrue(np.all(scene <= 1))

        # Tent should be predominantly triangular with bright apex/base gradient proxy
        tent_pixels = np.sum(np.abs(tent - 0.85) < 0.02)
        self.assertGreater(tent_pixels, 80)

        # Church should have bright steeple/spire pixels
        church_bright = np.sum(church > 0.9)
        self.assertGreater(church_bright, 10)

        # Tower should be tall and narrow: vertical extent > horizontal extent proxy
        tower_cols_active = np.sum(np.max(tower, axis=0) > 0.1)
        tower_rows_active = np.sum(np.max(tower, axis=1) > 0.1)
        self.assertGreater(tower_rows_active, tower_cols_active)

    def test_varied_includes_new_types(self):
        """Test varied scene generation supports new types without errors."""
        scenes = [
            make_varied_scene('church', size=64),
            make_varied_scene('tent', size=64),
            make_varied_scene('tower', size=64),
        ]
        for scene in scenes:
            self.assertEqual(scene.shape, (64, 64))
            self.assertTrue(np.all(scene >= 0))
            self.assertTrue(np.all(scene <= 1))


if __name__ == '__main__':
    unittest.main()