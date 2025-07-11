import tensorflow as tf

# Define dummy class names
class_names = ['plastic', 'organic', 'metal']
# Load MobileNetV2 without top layer
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save dummy model
model.save("model/waste_model.h5")
print("✅ Dummy model saved to model/waste_model.h5")
