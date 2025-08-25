import json
from datetime import timedelta

from cooperative_cuisine import ROOT_DIR
from cooperative_cuisine.environment import Environment

from cocu_base_agents.new_agent.new_agent import NewAgent

if __name__ == "__main__":
    env = Environment(
        env_config=ROOT_DIR / "configs" / "environment_config.yaml",
        layout_config=ROOT_DIR / "configs" / "layouts" / "basic.layout",
        item_info=ROOT_DIR / "configs" / "item_info.yaml"
    )
    env.add_player("0")
    
    env.env_time_end = env.env_time + timedelta(
        seconds=20
    )
    
    recipe_graphs = env.recipe_validation.get_recipe_graphs()
    
    agent2 = NewAgent("0", 0.2, json.dumps(recipe_graphs), "vc_url", "vc_room", additional_coroutines=[])
    agent2.run_via_env_reference(env)