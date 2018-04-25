from yolo.net.net import Net
import tensorflow as tf


class YOLOTinyD(Net):
    def __init__(self, common_params, net_params, test=False):
        """
                common params: a params dict
                net_params   : a params dict
                """
        super(YOLOTinyD, self).__init__(common_params, net_params)
        # process params
        self.image_size = int(common_params['image_size'])
        self.num_classes = int(common_params['num_classes'])
        self.cell_size = int(net_params['cell_size'])
        self.boxes_per_cell = int(net_params['boxes_per_cell'])
        self.batch_size = int(common_params['batch_size'])
        self.weight_decay = float(net_params['weight_decay'])

        if not test:
            self.object_scale = float(net_params['object_scale'])
            self.noobject_scale = float(net_params['noobject_scale'])
            self.class_scale = float(net_params['class_scale'])
            self.coord_scale = float(net_params['coord_scale'])

    def inference(self, images, croped_images):
        """Build the yolo model

                Args:
                  images:  4-D tensor [batch_size, image_height, image_width, channels]
                Returns:
                  predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
                """

        # croped_images = self.crop_labels(images, labels)
        # croped image  convolutioned
        with  tf.name_scope('convolution_croped_image_d'):
            conv_num_croped = 1

            temp_conv_croped = self.conv2d('conv_croped' + str(conv_num_croped), croped_images, [3, 3, 3, 16], stride=1)

            conv_num_croped += 1

            temp_pool_croped = self.max_pool(temp_conv_croped, [2, 2], 2)

            temp_conv_croped = self.conv2d('conv_croped' + str(conv_num_croped), temp_pool_croped, [3, 3, 16, 32], stride=1)

            conv_num_croped += 1

            temp_pool_croped = self.max_pool(temp_conv_croped, [2, 2], 2)

            temp_conv_croped = self.conv2d('conv_croped' + str(conv_num_croped), temp_pool_croped, [3, 3, 32, 64], stride=1)
            conv_num_croped += 1

            temp_conv_croped = self.max_pool(temp_conv_croped, [2, 2], 2)

            temp_conv_croped = self.conv2d('conv_croped' + str(conv_num_croped), temp_conv_croped, [3, 3, 64, 128], stride=1)
            conv_num_croped += 1

            temp_conv_croped = self.max_pool(temp_conv_croped, [2, 2], 2)

            temp_conv_croped = self.conv2d('conv_croped' + str(conv_num_croped), temp_conv_croped, [3, 3, 128, 256], stride=1)
            conv_num_croped += 1

            temp_conv_croped = self.max_pool(temp_conv_croped, [2, 2], 2)

            temp_conv_croped = self.conv2d('conv_croped' + str(conv_num_croped), temp_conv_croped, [3, 3, 256, 512], stride=1)
            conv_num_croped += 1

            temp_conv_croped = self.max_pool(temp_conv_croped, [2, 2], 2)
        # temp_conv_croped = self.conv2d('conv' + str(conv_num_croped), temp_conv_croped, [3, 3, 512, 1024], stride=1)
        # conv_num_croped += 1

        # temp_conv_croped = self.conv2d('conv' + str(conv_num_croped), temp_conv_croped, [3, 3, 1024, 1024], stride=1)
        # conv_num_croped += 1

        # temp_conv_croped = self.conv2d('conv' + str(conv_num_croped), temp_conv_croped, [3, 3, 1024, 1024], stride=1)
        # conv_num_croped += 1

        # temp_conv_croped = tf.transpose(temp_conv_croped, (0, 3, 1, 2))


        # original images convolutioned
        with tf.name_scope('convolution_oimage_d'):
            conv_num = 1

            temp_conv = self.conv2d('conv_d' + str(conv_num), images, [3, 3, 3, 16], stride=1)
            conv_num += 1

            temp_pool = self.max_pool(temp_conv, [2, 2], 2)

            temp_conv = self.conv2d('conv_d' + str(conv_num), temp_pool, [3, 3, 16, 32], stride=1)
            conv_num += 1

            temp_pool = self.max_pool(temp_conv, [2, 2], 2)

            temp_conv = self.conv2d('conv_d' + str(conv_num), temp_pool, [3, 3, 32, 64], stride=1)
            conv_num += 1

            temp_conv = self.max_pool(temp_conv, [2, 2], 2)

            temp_conv = self.conv2d('conv_d' + str(conv_num), temp_conv, [3, 3, 64, 128], stride=1)
            conv_num += 1

            temp_conv = self.max_pool(temp_conv, [2, 2], 2)

            temp_conv = self.conv2d('conv_d' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1)
            conv_num += 1

            temp_conv = self.max_pool(temp_conv, [2, 2], 2)

            temp_conv = self.conv2d('conv_d' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
            conv_num += 1

            temp_conv = self.max_pool(temp_conv, [2, 2], 2)

        # temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
        # conv_num += 1

        # temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
        # conv_num += 1

        # temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
        # conv_num += 1

        # temp_conv = tf.transpose(temp_conv, (0, 3, 1, 2))

        # concat the two features  and then fully connected
        temp_conv_mergin = tf.concat([temp_conv, temp_conv_croped], axis=3)

        # Fully connected layer
        local1 = self.local('local1_d', temp_conv_mergin, self.cell_size * self.cell_size * 1024, 256)

        local2 = self.local('local2_d', local1, 256, 4096)

        local3 = self.local('local3_d', local2, 4096, 1, leaky=False, pretrain=False, train=True)

        return local3

    def loss(self, predicts, labels, objects_num):
        pass

    def loss_d(self, groundtruth, label_predicted):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=groundtruth, logits=label_predicted))


