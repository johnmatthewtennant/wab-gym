import wab_env
import unittest
import numpy as np


wrapper = wab_env.PragmaticObsWrapper(wab_env.WolvesAndBushesEnv())
class TestPragmaticObsWrapper(unittest.TestCase):

    def test_TwoEquidistantBushes(self):
        wolves, bushes, ostriches = np.zeros((11,11)), np.zeros((11,11)), np.zeros((11,11))
        bushes[6,3] = 1
        bushes[7,4] = 1
        bushes[8,6] = 1
        bushes[6,10] = 1

        wolves[5,5] = 1
        wolves[6,6] = 1
        wolves[4,4] = 1

        tile_mask = np.array(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            ]
        )

        obs = wrapper.observation((wolves, bushes, ostriches, 40, 0, 0, tile_mask))

        obs2 = ([0,0,0,0],
         [0, 10, 10, 0],
         np.asarray([1, 1, 1, 1]),
         [0, 0, 9, 10],
         [0, 0, 10, 9],
         np.asarray([0, 2, 4, 2]),
         0,
         40,
         0,
         0,
         np.asarray([[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
               [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
               [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]])
         )
        self.assertEqual(obs[0], obs2[0])
        self.assertEqual(obs[1], obs2[1])
        self.assertEqual(obs[2].tolist(), obs2[2].tolist())
        self.assertEqual(obs[3], obs2[3])
        self.assertEqual(obs[4], obs2[4])
        self.assertEqual(obs[5].tolist(), obs2[5].tolist())

    def test_standing_on_bush(self):
        wolves, bushes, ostriches = np.zeros((11,11)), np.zeros((11,11)), np.zeros((11,11))
        bushes[5,5] = 1

        tile_mask = np.array(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            ]
        )

        obs = wrapper.observation((wolves, bushes, ostriches, 40, 0, 0, tile_mask))

        obs2 = ([0,0,0,0],
                [0, 0, 0, 0],
                np.asarray([0, 0, 0, 0]),
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                np.asarray([0, 0, 0, 0]),
                1,
                40,
                0,
                0,
                np.asarray([[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]])
                )
        self.assertEqual(obs[6], obs2[6])

    def test_numerous_bushes_and_wolves_with_blindspots(self):
        wolves, bushes, ostriches = np.zeros((11,11)), np.zeros((11,11)), np.zeros((11,11))

        wolves[2,:] = 1
        wolves[:, 6] = 1
        bushes[1, :] = 1
        bushes[9, :] = 1

        tile_mask = np.array(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            ]
        )

        wolves[np.where(tile_mask == 1)] = 0
        bushes[np.where(tile_mask == 1)] = 0

        obs = wrapper.observation((wolves, bushes, ostriches, 40, 0, 0, tile_mask))

        obs2 = ([0,10,0,0],
                [0, 10, 10, 0],
                np.asarray([10, 10, 5, 4]),
                [0, 0, 7, 0],
                [7, 0, 0, 0],
                np.asarray([7, 6, 7, 6]),
                0,
                40,
                0,
                0,
                np.asarray([[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]])
                )
        self.assertEqual(obs[0], obs2[0])
        self.assertEqual(obs[1], obs2[1])
        self.assertEqual(obs[2].tolist(), obs2[2].tolist())
        self.assertEqual(obs[3], obs2[3])
        self.assertEqual(obs[4], obs2[4])
        self.assertEqual(obs[5].tolist(), obs2[5].tolist())



if __name__ == "__main__":
    unittest.main()
