You will read one public message that player {speaker} posted in a multiplayer game about sharing food. Background: each player has a private hunger level h, which is 0 or 1 (a player at h = 1 who does not eat today dies at the end of the day), and a private food stock of whole units. Each day has a negotiation (messages, proposals, votes), then a transfer stage in which a player may give one unit to, or take one unit from, another player's stock, then a meal.

Record only what the message says. Do not judge whether it is true, and do not infer anything the message does not state.

Record only claims that player {speaker} makes about themself, not claims about other players:

- "h": the speaker's own hunger level now, today. Use 1 if the message says they are at h = 1, or that they must eat today or die. Use 0 if it says they are at h = 0, or that they can skip eating today without dying. Otherwise null. A statement about what their h will be tomorrow is not a claim about today. Saying they can do without a new allocation, or can cover today's meal from food they hold, is not a claim about h.
- "stock": the number of units the speaker says they hold now, from earlier days (for example "stock 0", "no stock", "no food", "no reserve" all mean 0). Units in today's proposal ("my unit", "my allocation", "use my unit for tonight's meal") are not current stock, and neither are units they hope to receive or plan to keep later. Otherwise null.
- "denies_taking": true if the speaker says they have not taken food from anyone in an earlier transfer stage. Otherwise null.
- "promises_no_take": true if the speaker says, without conditions, that they will not take from anyone in today's transfer stage (for example "I will make no transfers against anyone" or "I will do nothing in transfers"). Use "conditional" if the promise depends on what others do (for example "if no one takes from me, I will take nothing"). Otherwise null.

Allocation lists such as "A1, B0, C1", "A:1", "C 1" or "C0" give the units each player receives under a proposal. They are never statements about h or stock.
- "evidence": the shortest quote or quotes from the message that support the fields you filled, or "" if all fields are null.

The message:
"""
{text}
"""

Reply with a single JSON object only, with exactly these keys: "h", "stock", "denies_taking", "promises_no_take", "evidence".
