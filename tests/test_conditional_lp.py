import unittest
import numpy as np
from baseline.lp_conditional import solve_conditional_lp, apply_rank_one

class LPTests(unittest.TestCase):
    def test_separates_two_orthogonal_features(self):
        r=np.array([[1.,0.,0.],[2.,0.,0.]])
        p=np.array([[0.,1.,0.],[0.,2.,0.]])
        result=solve_conditional_lp(r,p,delta=1,epsilon=.05)
        self.assertLessEqual(np.max(r @ result.v),-1+1e-6)
        self.assertLessEqual(np.max(np.abs(p @ result.v)),.05+1e-6)
        self.assertAlmostEqual(result.v[0],-1,places=6)
        self.assertAlmostEqual(result.v[1],0,places=6)
        self.assertAlmostEqual(result.v[2],0,places=6)
    def test_edit_has_expected_token_effect(self):
        w=np.zeros((2,3))
        u=np.array([1.,0.])
        v=np.array([-1.,0.,0.])
        h=np.array([2.,3.,4.])
        new=apply_rank_one(w,u,v)
        self.assertTrue(np.allclose((new-w)@h,u*(v@h)))
    def test_infeasible_hard_constraints_resolved_by_slack(self):
        same=np.array([[1.,0.]])
        result=solve_conditional_lp(same,same,delta=1,epsilon=0)
        self.assertTrue(np.isfinite(result.objective))
        self.assertGreater(result.loop_slack.sum()+result.productive_slack.sum(),0)
    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            solve_conditional_lp(np.ones((3,2)),np.ones((4,3)))

if __name__=="__main__":
    unittest.main()
