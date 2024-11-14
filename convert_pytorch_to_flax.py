from transformers import FlaxAutoModel, AutoModel

# Load the PyTorch model from the directory
torch_model = AutoModel.from_pretrained("lwm_checkpoint", from_tf=False)

# Convert the model to Flax
flax_model = FlaxAutoModel.from_pretrained("lwm_checkpoint", from_pt=True)

# Save the Flax model to a new directory
flax_model.save_pretrained("lwm_checkpoint_converted_Flax")

print("Conversion complete. Flax model saved at 'lwm_checkpoint_converted_Flax'.")


