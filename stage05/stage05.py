with open("candidate_text.txt", "r") as f:
  candidate_text = f.read()

stripped_candidate_text = ''.join(e for e in candidate_text if e.isalnum())

with open("ciphertext_stage05.txt", "r") as f:
  ciphertext = f.read()
    
indicies = ciphertext.split()

decrypted_message = ""
for index in indicies:
  # The indicies in the ciphertext are 1-based, so we need to 
  # subtract 1 to get the correct index in the stripped candidate 
  # text.
  decrypted_message += stripped_candidate_text[int(index) - 1].lower()
    
print(f"Decrypted message: {decrypted_message}")
