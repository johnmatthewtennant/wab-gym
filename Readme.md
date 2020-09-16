Install

`pipenv install`


Start it like
`env = WolvesAndBushesEnv()`
then you call `env.step(selected_action)` on it over and over
it will return an object like `(observation, reward, done, debug_output)`
observation is a tuple like `(wolf_grid, bush_grid, food, role, alive_killed_starved)`

the avalable actions are:
0: up
1: right
2: down
3: left
4: stay
5: switch role
