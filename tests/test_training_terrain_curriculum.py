import unittest
from types import SimpleNamespace as NS
import torch
from pbfm.terrains.training_curriculum import maximum_level, sample_levels, apply_reset_curriculum


class CurriculumTests(unittest.TestCase):
    def core(self, step=192020480):
        terrain = NS(terrain_origins=torch.arange(10*7*3).reshape(10,7,3).float(),
                     terrain_levels=torch.full((32,),9), terrain_types=torch.arange(32)%7,
                     env_origins=torch.zeros(32,3))
        return NS(_creation_config=NS(terrain_training_curriculum=True,
                  terrain_curriculum_interval=19200000, terrain_curriculum_start_global_step=192020480,
                  terrain_curriculum_easy_fraction=.25), _training_terrain_global_step=step,
                  terrain_enabled=True, terrain_mode='training_mix', device='cpu',
                  mjlab_env=NS(scene={'terrain':terrain}), env_origins=torch.zeros(32,3))

    def test_schedule_boundaries(self):
        for step, expected in [(0,2),(19199999,2),(19200000,4),(38400000,6),(57600000,8),(76800000,9),(999999999,9)]:
            self.assertEqual(maximum_level(step,10,19200000),expected)

    def test_invalid_inputs(self):
        for args in [(-1,10,1),(0,0,1),(0,10,0)]:
            with self.assertRaises(ValueError): maximum_level(*args)

    def test_resume_relative_clock_and_only_requested_envs(self):
        c=self.core();ids=torch.tensor([0,3,9]);t=c.mjlab_env.scene['terrain'];types=t.terrain_types.clone()
        apply_reset_curriculum(c,ids)
        self.assertTrue((t.terrain_levels[ids]<=2).all())
        self.assertEqual(t.terrain_levels[1].item(),9)
        torch.testing.assert_close(t.terrain_types,types)
        torch.testing.assert_close(c.env_origins[ids],t.terrain_origins[t.terrain_levels[ids],types[ids]])
        torch.testing.assert_close(t.env_origins,c.env_origins)

    def test_inference_not_armed(self):
        c=self.core(None);apply_reset_curriculum(c,torch.arange(32))
        self.assertTrue((c.mjlab_env.scene['terrain'].terrain_levels==9).all())

    def test_target_state_reset_unchanged(self):
        c=self.core();apply_reset_curriculum(c,torch.arange(32),{'root_states':None})
        self.assertTrue((c.mjlab_env.scene['terrain'].terrain_levels==9).all())

    def test_disabled_unchanged(self):
        c=self.core();c._creation_config.terrain_training_curriculum=False
        apply_reset_curriculum(c,torch.arange(32))
        self.assertTrue((c.mjlab_env.scene['terrain'].terrain_levels==9).all())

    def test_easy_mixture(self):
        torch.manual_seed(4)
        v=sample_levels(100000,device='cpu',maximum=9,easy_fraction=.25)
        # 25% explicitly easy + 75% uniform all levels => 47.5% low levels.
        self.assertAlmostEqual((v<=2).float().mean().item(),.475,delta=.01)
        self.assertEqual(v.max().item(),9)

    def test_easy_only(self):
        v=sample_levels(1000,device='cpu',maximum=9,easy_fraction=1.)
        self.assertTrue((v<=2).all())

if __name__=='__main__': unittest.main()
