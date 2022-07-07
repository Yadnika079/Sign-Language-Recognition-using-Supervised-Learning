#!/usr/bin/env python
# coding: utf-8

# In[15]:


#importing modules
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib

matplotlib.style.use('ggplot')


# In[16]:


#data generation


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_generator = datagen.flow_from_directory(
    directory=r"C:\Users\asua\Documents\ASL\asl_alphabet_train",
    target_size=(200, 200),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = datagen.flow_from_directory(
    directory=r"C:\Users\asua\Documents\ASL\asl_alphabet_val",
    target_size=(200, 200),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)




# # Model 1: SIMPLE CONVNET+MAX POOLING (BASE MODEL)

# In[17]:


#model definition

def build_model(num_classes):
    model = tf.keras.Sequential([
    
    
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',input_shape= (200,200,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
   
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


# In[18]:


model1 = build_model(num_classes=29)


# In[19]:


#compiling
model1.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# In[20]:


#checking number of parameters
print(model1.summary())


# In[21]:


EPOCHS = 15
BATCH_SIZE = 128
history1 = model1.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE,
                    verbose=1
                    )


# In[22]:


train_loss1 = history1.history['loss']
train_acc1 = history1.history['accuracy']
valid_loss1 = history1.history['val_loss']
valid_acc1 = history1.history['val_accuracy']


# In[24]:


def save_plots(train_acc1, valid_acc1, train_loss1, valid_loss1):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc1, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc1, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    # loss plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss1, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss1, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

save_plots(train_acc1, valid_acc1, train_loss1, valid_loss1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Model 2 : CONVNET+MAXPOOL+DROPOUT (IN FC LAYERS)

# In[ ]:





# In[25]:


#model definition

def build_model(num_classes):
    model2 = tf.keras.Sequential([
   
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',input_shape= (200,200,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model2


# In[26]:


model2 = build_model(num_classes=29)


# In[27]:


#compiling
model2.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# In[28]:


#checking number of parameters
print(model2.summary())


# In[29]:


EPOCHS = 20
BATCH_SIZE = 128
history2 = model2.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE,
                    verbose=1
                    )


# In[30]:


train_loss_2 = history2.history['loss']
train_acc_2 = history2.history['accuracy']
valid_loss_2 = history2.history['val_loss']
valid_acc_2 = history2.history['val_accuracy']


# In[31]:


def save_plots(train_acc_2, valid_acc_2, train_loss_2, valid_loss_2):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc_2, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc_2, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy for dropout in FC layers')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    # loss plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss_2, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss_2, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss for dropout in Fc layers')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

save_plots(train_acc_2, valid_acc_2, train_loss_2, valid_loss_2)


# In[ ]:





# # Model 3 : CONVNET+AVERAGE POOLING

# In[32]:


#model definition

def build_model(num_classes):
    model3 = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',input_shape= (200,200,3)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation= 'relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model3


model3 = build_model(num_classes=29)


# In[33]:


#compiling
model3.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# In[34]:


#checking number of parameters
print(model3.summary())


# In[35]:


EPOCHS = 20
BATCH_SIZE = 128
history3 = model3.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE,
                    verbose=1
                    )


# In[36]:


train_loss_3 = history3.history['loss']
train_acc_3 = history3.history['accuracy']
valid_loss_3 = history3.history['val_loss']
valid_acc_3 = history3.history['val_accuracy']


# In[37]:


def save_plots(train_acc_3, valid_acc_3, train_loss_3, valid_loss_3):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc_3, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc_3, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    # loss plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss_3, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss_3, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

save_plots(train_acc_3, valid_acc_3, train_loss_3, valid_loss_3)


# In[ ]:





# # Model 4: CONVNET+MAXPOOL+DROPOUT (IN ALL LAYERS)

# In[38]:


#model definition

def build_model(num_classes):
    model4 = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',input_shape= (200,200,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.15),
    
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.15),
        
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.15),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model4


model4 = build_model(num_classes=29)


# In[39]:


#compiling
model4.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# In[40]:


#checking number of parameters
print(model4.summary())


# In[41]:


EPOCHS = 20
BATCH_SIZE = 128
history4 = model4.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE,
                    verbose=1
                    )


# In[42]:


train_loss_4 = history4.history['loss']
train_acc_4 = history4.history['accuracy']
valid_loss_4 = history4.history['val_loss']
valid_acc_4 = history4.history['val_accuracy']


# In[43]:


def save_plots(train_acc_4, valid_acc_4, train_loss_4, valid_loss_4):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc_4, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc_4, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy for dropout in all layers')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    # loss plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss_4, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss_4, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss for dropout in all layers')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

save_plots(train_acc_4, valid_acc_4, train_loss_4, valid_loss_4)


# In[ ]:





# In[ ]:





# In[ ]:





# # Model 5 :  ADDING L2 REGULARIZATION
# 

# In[44]:




#model definition

def build_model(num_classes):
    model5 = tf.keras.Sequential([
    
    
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',input_shape= (200,200,3),kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model5


model5 = build_model(num_classes=29)


# In[45]:


#compiling
model5.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# In[46]:


#checking number of parameters
print(model5.summary())


# In[47]:


EPOCHS = 20
BATCH_SIZE = 128
history5 = model5.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE,
                    verbose=1
                    )


# In[48]:


train_loss_5 = history5.history['loss']
train_acc_5 = history5.history['accuracy']
valid_loss_5 = history5.history['val_loss']
valid_acc_5 = history5.history['val_accuracy']


# In[49]:


def save_plots(train_acc_5, valid_acc_5, train_loss_5, valid_loss_5):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc_5, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc_5, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    # loss plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss_5, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss_5, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

save_plots(train_acc_5, valid_acc_5, train_loss_5, valid_loss_5)


# In[ ]:





# In[ ]:





# In[ ]:




