import copy

def clones(module, no_of_copies):
  """Produce no_of_copies identical layers."""
  return nn.ModuleList([copy.deepcopy(module) for _ in range(no_of_copies)])

def convert_to_unicode(text):
  if isinstance(text, str):
    return text
  elif isinstance(text, bytes):
    return text.decode('utf-8')
  else:
    raise ValueError('Unsupported string type: %s' % (type(text)))

def truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

