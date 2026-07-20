from anthropic import Anthropic
client = Anthropic()
for entry in client.messages.batches.results("msgbatch_019QrH3JUnrePQvG4KcAv9jm"):
    print(entry.custom_id, entry.result.type)
    if entry.result.type == "errored":
        print(entry.result.error)   # the real reason
        break   # they're all the same; one is enough