
print('正在创建模型')
inputs = Input(shape=(sequence_length,),dtype='int32')
embedding = Embedding(input_dim=vocabulary_size,output_dim=embedding_dim,input_length=sequence_lengrh)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters,kernel_size=(filter_size[0],embedding_dim),padding='valid',kernel_ini
