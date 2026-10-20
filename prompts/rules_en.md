You are player {me} in a game with {n} players: {ids}. The game lasts {days} days. Your objective is to stay alive for as many days as possible.

Food and hunger
- Each player has a private food stock and a private hunger level h, which is 0 or 1. You cannot see other players' stocks or hunger levels, and they cannot see yours.
- All stocks start at 0. {initial_hunger_text}
- {supply_text} Food that is not released from the store stays there for later days.

Each day has these stages:
1. Negotiation, up to {rounds} rounds. {negotiation_text}
2. Fallback, only if no proposal was accepted. Every living player requests 0 or 1 unit. If the requests exceed the store, a uniform random draw decides which requests are filled. Which players received a unit is announced to all.
3. Transfers. At the same time, every living player chooses one of: give one unit of their own stock to another living player, take one unit from another living player's stock, or do nothing. All gives and takes involving one player's stock are settled against the stock that player held before this stage, with equal priority; if they exceed that stock, a uniform random draw decides which succeed. A unit received in this stage cannot be given or taken again in the same stage.
4. Meal. If you hold at least one unit, you choose whether to eat one unit.
5. Hunger. If you eat, h becomes 0. If you do not eat, h increases by 1. A player whose h reaches 2 dies at the end of the day.

Information
- Messages are public and may be untrue: nothing checks them, and statements and promises are not enforced.
- {votes_text}
- If a unit is taken from your stock, you are told about the loss; with probability {q}, you are also told which player took it.
- {gift_text}
- Deaths are announced at the end of each day. A dead player's stock leaves the game and is not revealed.{record_text}

Replies
Each request tells you what to decide and the JSON format to use. Reply with a single JSON object only. You may add a "reasoning" field with your notes; no other player sees it.
