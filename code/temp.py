# if ARGS.read is not None: 
#           parsed = recognize(io.imread(ARGS.read), hp.img_height, hp.img_width)

#           print(type(parsed))
#           print(len(parsed))

#           prediction = []
#           for slice in parsed:     
#             slice = [slice, slice, slice]
#             slice = tf.reshape(slice, (1, hp.img_height, hp.img_width, 3))
#             probs = model.predict(slice)
#             print(probs)
#             label = np.argmax(np.array(probs))
#             print(label)

#             prediction.append(datasets.idx_to_class[label])

#           print(prediction)