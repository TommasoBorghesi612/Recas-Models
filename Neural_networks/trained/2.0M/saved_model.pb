І№
§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02unknown8иа

model/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namemodel/conv1/kernel

&model/conv1/kernel/Read/ReadVariableOpReadVariableOpmodel/conv1/kernel*&
_output_shapes
:*
dtype0
x
model/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namemodel/conv1/bias
q
$model/conv1/bias/Read/ReadVariableOpReadVariableOpmodel/conv1/bias*
_output_shapes
:*
dtype0

model/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namemodel/conv2/kernel

&model/conv2/kernel/Read/ReadVariableOpReadVariableOpmodel/conv2/kernel*&
_output_shapes
: *
dtype0
x
model/conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namemodel/conv2/bias
q
$model/conv2/bias/Read/ReadVariableOpReadVariableOpmodel/conv2/bias*
_output_shapes
: *
dtype0

model/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_namemodel/conv3/kernel

&model/conv3/kernel/Read/ReadVariableOpReadVariableOpmodel/conv3/kernel*&
_output_shapes
:  *
dtype0
x
model/conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namemodel/conv3/bias
q
$model/conv3/bias/Read/ReadVariableOpReadVariableOpmodel/conv3/bias*
_output_shapes
: *
dtype0

model/conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_namemodel/conv4/kernel

&model/conv4/kernel/Read/ReadVariableOpReadVariableOpmodel/conv4/kernel*&
_output_shapes
: @*
dtype0
x
model/conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namemodel/conv4/bias
q
$model/conv4/bias/Read/ReadVariableOpReadVariableOpmodel/conv4/bias*
_output_shapes
:@*
dtype0

model/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namemodel/dense1/kernel
}
'model/dense1/kernel/Read/ReadVariableOpReadVariableOpmodel/dense1/kernel* 
_output_shapes
:
*
dtype0
{
model/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namemodel/dense1/bias
t
%model/dense1/bias/Read/ReadVariableOpReadVariableOpmodel/dense1/bias*
_output_shapes	
:*
dtype0

model/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*$
shared_namemodel/dense2/kernel
|
'model/dense2/kernel/Read/ReadVariableOpReadVariableOpmodel/dense2/kernel*
_output_shapes
:	@*
dtype0
z
model/dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namemodel/dense2/bias
s
%model/dense2/bias/Read/ReadVariableOpReadVariableOpmodel/dense2/bias*
_output_shapes
:@*
dtype0

model/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_namemodel/output/kernel
{
'model/output/kernel/Read/ReadVariableOpReadVariableOpmodel/output/kernel*
_output_shapes

:@*
dtype0
z
model/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namemodel/output/bias
s
%model/output/bias/Read/ReadVariableOpReadVariableOpmodel/output/bias*
_output_shapes
:*
dtype0
{
model/norm1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*"
shared_namemodel/norm1/gamma
t
%model/norm1/gamma/Read/ReadVariableOpReadVariableOpmodel/norm1/gamma*
_output_shapes	
:Р*
dtype0
y
model/norm1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*!
shared_namemodel/norm1/beta
r
$model/norm1/beta/Read/ReadVariableOpReadVariableOpmodel/norm1/beta*
_output_shapes	
:Р*
dtype0

model/norm1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*(
shared_namemodel/norm1/moving_mean

+model/norm1/moving_mean/Read/ReadVariableOpReadVariableOpmodel/norm1/moving_mean*
_output_shapes	
:Р*
dtype0

model/norm1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*,
shared_namemodel/norm1/moving_variance

/model/norm1/moving_variance/Read/ReadVariableOpReadVariableOpmodel/norm1/moving_variance*
_output_shapes	
:Р*
dtype0
z
model/norm2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*"
shared_namemodel/norm2/gamma
s
%model/norm2/gamma/Read/ReadVariableOpReadVariableOpmodel/norm2/gamma*
_output_shapes
:[*
dtype0
x
model/norm2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*!
shared_namemodel/norm2/beta
q
$model/norm2/beta/Read/ReadVariableOpReadVariableOpmodel/norm2/beta*
_output_shapes
:[*
dtype0

model/norm2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*(
shared_namemodel/norm2/moving_mean

+model/norm2/moving_mean/Read/ReadVariableOpReadVariableOpmodel/norm2/moving_mean*
_output_shapes
:[*
dtype0

model/norm2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*,
shared_namemodel/norm2/moving_variance

/model/norm2/moving_variance/Read/ReadVariableOpReadVariableOpmodel/norm2/moving_variance*
_output_shapes
:[*
dtype0

NoOpNoOp
4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*й3
valueЯ3BЬ3 BХ3
х
	conv1
	conv2
	pool1
	conv3
	conv4
	pool2
flatten
d1
	d2

d3
b_norm_1
b_norm_2
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
R
.regularization_losses
/	variables
0trainable_variables
1	keras_api
R
2regularization_losses
3	variables
4trainable_variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
h

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
h

Bkernel
Cbias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api

Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api

Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
 
І
0
1
2
3
"4
#5
(6
)7
68
79
<10
=11
B12
C13
I14
J15
K16
L17
R18
S19
T20
U21

0
1
2
3
"4
#5
(6
)7
68
79
<10
=11
B12
C13
I14
J15
R16
S17
­
Znon_trainable_variables
[layer_metrics

\layers
regularization_losses
	variables
trainable_variables
]metrics
^layer_regularization_losses
 
OM
VARIABLE_VALUEmodel/conv1/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmodel/conv1/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
_non_trainable_variables
`layer_metrics

alayers
regularization_losses
	variables
trainable_variables
bmetrics
clayer_regularization_losses
OM
VARIABLE_VALUEmodel/conv2/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmodel/conv2/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
dnon_trainable_variables
elayer_metrics

flayers
regularization_losses
	variables
trainable_variables
gmetrics
hlayer_regularization_losses
 
 
 
­
inon_trainable_variables
jlayer_metrics

klayers
regularization_losses
	variables
 trainable_variables
lmetrics
mlayer_regularization_losses
OM
VARIABLE_VALUEmodel/conv3/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmodel/conv3/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
­
nnon_trainable_variables
olayer_metrics

players
$regularization_losses
%	variables
&trainable_variables
qmetrics
rlayer_regularization_losses
OM
VARIABLE_VALUEmodel/conv4/kernel'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmodel/conv4/bias%conv4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
­
snon_trainable_variables
tlayer_metrics

ulayers
*regularization_losses
+	variables
,trainable_variables
vmetrics
wlayer_regularization_losses
 
 
 
­
xnon_trainable_variables
ylayer_metrics

zlayers
.regularization_losses
/	variables
0trainable_variables
{metrics
|layer_regularization_losses
 
 
 
Џ
}non_trainable_variables
~layer_metrics

layers
2regularization_losses
3	variables
4trainable_variables
metrics
 layer_regularization_losses
MK
VARIABLE_VALUEmodel/dense1/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEmodel/dense1/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
В
non_trainable_variables
layer_metrics
layers
8regularization_losses
9	variables
:trainable_variables
metrics
 layer_regularization_losses
MK
VARIABLE_VALUEmodel/dense2/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEmodel/dense2/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
В
non_trainable_variables
layer_metrics
layers
>regularization_losses
?	variables
@trainable_variables
metrics
 layer_regularization_losses
MK
VARIABLE_VALUEmodel/output/kernel$d3/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEmodel/output/bias"d3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
В
non_trainable_variables
layer_metrics
layers
Dregularization_losses
E	variables
Ftrainable_variables
metrics
 layer_regularization_losses
 
PN
VARIABLE_VALUEmodel/norm1/gamma)b_norm_1/gamma/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEmodel/norm1/beta(b_norm_1/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEmodel/norm1/moving_mean/b_norm_1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodel/norm1/moving_variance3b_norm_1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1
K2
L3

I0
J1
В
non_trainable_variables
layer_metrics
layers
Mregularization_losses
N	variables
Otrainable_variables
metrics
 layer_regularization_losses
 
PN
VARIABLE_VALUEmodel/norm2/gamma)b_norm_2/gamma/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEmodel/norm2/beta(b_norm_2/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEmodel/norm2/moving_mean/b_norm_2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodel/norm2/moving_variance3b_norm_2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1
T2
U3

R0
S1
В
non_trainable_variables
layer_metrics
layers
Vregularization_losses
W	variables
Xtrainable_variables
metrics
 layer_regularization_losses

K0
L1
T2
U3
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

K0
L1
 
 
 
 

T0
U1
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:џџџџџџџџџл *
dtype0*
shape:џџџџџџџџџл 

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1model/conv1/kernelmodel/conv1/biasmodel/conv2/kernelmodel/conv2/biasmodel/conv3/kernelmodel/conv3/biasmodel/conv4/kernelmodel/conv4/biasmodel/norm1/moving_meanmodel/norm1/moving_variancemodel/norm1/betamodel/norm1/gammamodel/norm2/moving_meanmodel/norm2/moving_variancemodel/norm2/betamodel/norm2/gammamodel/dense1/kernelmodel/dense1/biasmodel/dense2/kernelmodel/dense2/biasmodel/output/kernelmodel/output/bias*"
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference_signature_wrapper_114847
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&model/conv1/kernel/Read/ReadVariableOp$model/conv1/bias/Read/ReadVariableOp&model/conv2/kernel/Read/ReadVariableOp$model/conv2/bias/Read/ReadVariableOp&model/conv3/kernel/Read/ReadVariableOp$model/conv3/bias/Read/ReadVariableOp&model/conv4/kernel/Read/ReadVariableOp$model/conv4/bias/Read/ReadVariableOp'model/dense1/kernel/Read/ReadVariableOp%model/dense1/bias/Read/ReadVariableOp'model/dense2/kernel/Read/ReadVariableOp%model/dense2/bias/Read/ReadVariableOp'model/output/kernel/Read/ReadVariableOp%model/output/bias/Read/ReadVariableOp%model/norm1/gamma/Read/ReadVariableOp$model/norm1/beta/Read/ReadVariableOp+model/norm1/moving_mean/Read/ReadVariableOp/model/norm1/moving_variance/Read/ReadVariableOp%model/norm2/gamma/Read/ReadVariableOp$model/norm2/beta/Read/ReadVariableOp+model/norm2/moving_mean/Read/ReadVariableOp/model/norm2/moving_variance/Read/ReadVariableOpConst*#
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*(
f#R!
__inference__traced_save_115823
г
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemodel/conv1/kernelmodel/conv1/biasmodel/conv2/kernelmodel/conv2/biasmodel/conv3/kernelmodel/conv3/biasmodel/conv4/kernelmodel/conv4/biasmodel/dense1/kernelmodel/dense1/biasmodel/dense2/kernelmodel/dense2/biasmodel/output/kernelmodel/output/biasmodel/norm1/gammamodel/norm1/betamodel/norm1/moving_meanmodel/norm1/moving_variancemodel/norm2/gammamodel/norm2/betamodel/norm2/moving_meanmodel/norm2/moving_variance*"
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__traced_restore_115901ўй

ь
Њ
B__inference_dense1_layer_call_and_return_conditional_losses_114490

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
л
{
&__inference_conv3_layer_call_fn_114047

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_1140372
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ч
Г
&__inference_model_layer_call_fn_115495
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1147002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџл 
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
А

Љ
A__inference_conv2_layer_call_and_return_conditional_losses_114003

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
њ
B
&__inference_pool1_layer_call_fn_114025

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_pool1_layer_call_and_return_conditional_losses_1140192
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ю
A__inference_norm1_layer_call_and_return_conditional_losses_115622

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Р2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Р2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:Р2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Р2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Р2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџР:::::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
л
{
&__inference_conv4_layer_call_fn_114069

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_1140592
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ђ

&__inference_norm1_layer_call_fn_115635

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_norm1_layer_call_and_return_conditional_losses_1141772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџР::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
ю
A__inference_norm2_layer_call_and_return_conditional_losses_115704

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:[2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:[2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ[:::::O K
'
_output_shapes
:џџџџџџџџџ[
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Щ)
Ў
A__inference_norm2_layer_call_and_return_conditional_losses_114317

inputs
assignmovingavg_114292
assignmovingavg_1_114298 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:[2
moments/StopGradientЄ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indicesВ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/114292*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_114292*
_output_shapes
:[*
dtype02 
AssignMovingAvg/ReadVariableOpУ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/114292*
_output_shapes
:[2
AssignMovingAvg/subК
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/114292*
_output_shapes
:[2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_114292AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/114292*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЄ
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/114298*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_114298*
_output_shapes
:[*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЭ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/114298*
_output_shapes
:[2
AssignMovingAvg_1/subФ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/114298*
_output_shapes
:[2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_114298AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/114298*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:[2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:[2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
batchnorm/add_1Е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ[::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:џџџџџџџџџ[
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Џ

Љ
A__inference_conv3_layer_call_and_return_conditional_losses_114037

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Щ)
Ў
A__inference_norm2_layer_call_and_return_conditional_losses_115684

inputs
assignmovingavg_115659
assignmovingavg_1_115665 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:[2
moments/StopGradientЄ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indicesВ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/115659*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_115659*
_output_shapes
:[*
dtype02 
AssignMovingAvg/ReadVariableOpУ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/115659*
_output_shapes
:[2
AssignMovingAvg/subК
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/115659*
_output_shapes
:[2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_115659AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/115659*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЄ
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/115665*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_115665*
_output_shapes
:[*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЭ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/115665*
_output_shapes
:[2
AssignMovingAvg_1/subФ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/115665*
_output_shapes
:[2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_115665AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/115665*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:[2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:[2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
batchnorm/add_1Е
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ[::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:џџџџџџџџџ[
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ:
ђ	
__inference__traced_save_115823
file_prefix1
-savev2_model_conv1_kernel_read_readvariableop/
+savev2_model_conv1_bias_read_readvariableop1
-savev2_model_conv2_kernel_read_readvariableop/
+savev2_model_conv2_bias_read_readvariableop1
-savev2_model_conv3_kernel_read_readvariableop/
+savev2_model_conv3_bias_read_readvariableop1
-savev2_model_conv4_kernel_read_readvariableop/
+savev2_model_conv4_bias_read_readvariableop2
.savev2_model_dense1_kernel_read_readvariableop0
,savev2_model_dense1_bias_read_readvariableop2
.savev2_model_dense2_kernel_read_readvariableop0
,savev2_model_dense2_bias_read_readvariableop2
.savev2_model_output_kernel_read_readvariableop0
,savev2_model_output_bias_read_readvariableop0
,savev2_model_norm1_gamma_read_readvariableop/
+savev2_model_norm1_beta_read_readvariableop6
2savev2_model_norm1_moving_mean_read_readvariableop:
6savev2_model_norm1_moving_variance_read_readvariableop0
,savev2_model_norm2_gamma_read_readvariableop/
+savev2_model_norm2_beta_read_readvariableop6
2savev2_model_norm2_moving_mean_read_readvariableop:
6savev2_model_norm2_moving_variance_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c69aeb140e314863b8977a2e23481c43/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Љ
valueBB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d3/bias/.ATTRIBUTES/VARIABLE_VALUEB)b_norm_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB(b_norm_1/beta/.ATTRIBUTES/VARIABLE_VALUEB/b_norm_1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3b_norm_1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)b_norm_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB(b_norm_2/beta/.ATTRIBUTES/VARIABLE_VALUEB/b_norm_2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3b_norm_2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesД
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesй	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_model_conv1_kernel_read_readvariableop+savev2_model_conv1_bias_read_readvariableop-savev2_model_conv2_kernel_read_readvariableop+savev2_model_conv2_bias_read_readvariableop-savev2_model_conv3_kernel_read_readvariableop+savev2_model_conv3_bias_read_readvariableop-savev2_model_conv4_kernel_read_readvariableop+savev2_model_conv4_bias_read_readvariableop.savev2_model_dense1_kernel_read_readvariableop,savev2_model_dense1_bias_read_readvariableop.savev2_model_dense2_kernel_read_readvariableop,savev2_model_dense2_bias_read_readvariableop.savev2_model_output_kernel_read_readvariableop,savev2_model_output_bias_read_readvariableop,savev2_model_norm1_gamma_read_readvariableop+savev2_model_norm1_beta_read_readvariableop2savev2_model_norm1_moving_mean_read_readvariableop6savev2_model_norm1_moving_variance_read_readvariableop,savev2_model_norm2_gamma_read_readvariableop+savev2_model_norm2_beta_read_readvariableop2savev2_model_norm2_moving_mean_read_readvariableop6savev2_model_norm2_moving_variance_read_readvariableop"/device:CPU:0*
_output_shapes
 *$
dtypes
22
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*с
_input_shapesЯ
Ь: ::: : :  : : @:@:
::	@:@:@::Р:Р:Р:Р:[:[:[:[: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&	"
 
_output_shapes
:
:!


_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::!

_output_shapes	
:Р:!

_output_shapes	
:Р:!

_output_shapes	
:Р:!

_output_shapes	
:Р: 

_output_shapes
:[: 

_output_shapes
:[: 

_output_shapes
:[: 

_output_shapes
:[:

_output_shapes
: 
№

&__inference_norm2_layer_call_fn_115730

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ[*&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_norm2_layer_call_and_return_conditional_losses_1143502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ[::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ[
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њ
ю
A__inference_norm2_layer_call_and_return_conditional_losses_114350

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:[2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:[2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ[:::::O K
'
_output_shapes
:џџџџџџџџџ[
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
є

&__inference_norm1_layer_call_fn_115648

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР*&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_norm1_layer_call_and_return_conditional_losses_1142102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџР::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ђb

A__inference_model_layer_call_and_return_conditional_losses_115397
input_1(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource&
"norm1_cast_readvariableop_resource(
$norm1_cast_1_readvariableop_resource(
$norm1_cast_2_readvariableop_resource(
$norm1_cast_3_readvariableop_resource&
"norm2_cast_readvariableop_resource(
$norm2_cast_1_readvariableop_resource(
$norm2_cast_2_readvariableop_resource(
$norm2_cast_3_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"   [   2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЎ
splitSplitVinput_1Const:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':џџџџџџџџџ :џџџџџџџџџ[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
ReshapeЇ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpР
conv1/Conv2DConv2DReshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1/Conv2D
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp 
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

conv1/ReluЇ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOpШ
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2/Conv2D
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp 
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

conv2/ReluЏ
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
pool1/MaxPoolЇ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOpХ
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv3/Conv2D
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp 
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

conv3/ReluЇ
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv4/Conv2D/ReadVariableOpЧ
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv4/Conv2D
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOp 
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

conv4/ReluЏ
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
pool2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten/Const
flatten/ReshapeReshapepool2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/Reshape
norm1/Cast/ReadVariableOpReadVariableOp"norm1_cast_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast/ReadVariableOp
norm1/Cast_1/ReadVariableOpReadVariableOp$norm1_cast_1_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast_1/ReadVariableOp
norm1/Cast_2/ReadVariableOpReadVariableOp$norm1_cast_2_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast_2/ReadVariableOp
norm1/Cast_3/ReadVariableOpReadVariableOp$norm1_cast_3_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast_3/ReadVariableOps
norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
norm1/batchnorm/add/y
norm1/batchnorm/addAddV2#norm1/Cast_1/ReadVariableOp:value:0norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/addv
norm1/batchnorm/RsqrtRsqrtnorm1/batchnorm/add:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/Rsqrt
norm1/batchnorm/mulMulnorm1/batchnorm/Rsqrt:y:0#norm1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/mul
norm1/batchnorm/mul_1Mulflatten/Reshape:output:0norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
norm1/batchnorm/mul_1
norm1/batchnorm/mul_2Mul!norm1/Cast/ReadVariableOp:value:0norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/mul_2
norm1/batchnorm/subSub#norm1/Cast_2/ReadVariableOp:value:0norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/sub
norm1/batchnorm/add_1AddV2norm1/batchnorm/mul_1:z:0norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
norm1/batchnorm/add_1
norm2/Cast/ReadVariableOpReadVariableOp"norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast/ReadVariableOp
norm2/Cast_1/ReadVariableOpReadVariableOp$norm2_cast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_1/ReadVariableOp
norm2/Cast_2/ReadVariableOpReadVariableOp$norm2_cast_2_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_2/ReadVariableOp
norm2/Cast_3/ReadVariableOpReadVariableOp$norm2_cast_3_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_3/ReadVariableOps
norm2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
norm2/batchnorm/add/y
norm2/batchnorm/addAddV2#norm2/Cast_1/ReadVariableOp:value:0norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
norm2/batchnorm/addu
norm2/batchnorm/RsqrtRsqrtnorm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/Rsqrt
norm2/batchnorm/mulMulnorm2/batchnorm/Rsqrt:y:0#norm2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul
norm2/batchnorm/mul_1Mulsplit:output:1norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
norm2/batchnorm/mul_1
norm2/batchnorm/mul_2Mul!norm2/Cast/ReadVariableOp:value:0norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul_2
norm2/batchnorm/subSub#norm2/Cast_2/ReadVariableOp:value:0norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/sub
norm2/batchnorm/add_1AddV2norm2/batchnorm/mul_1:z:0norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
norm2/batchnorm/add_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЄ
concatConcatV2norm1/batchnorm/add_1:z:0norm2/batchnorm/add_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ2
concatЄ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulconcat:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/MatMulЂ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/ReluЃ
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense2/MatMul/ReadVariableOp
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/MatMulЁ
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense2/BiasAdd/ReadVariableOp
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/ReluЂ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldense2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/Relum
IdentityIdentityoutput/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл :::::::::::::::::::::::Q M
(
_output_shapes
:џџџџџџџџџл 
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

D
(__inference_flatten_layer_call_fn_115506

inputs
identityЃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1143992
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
љ
|
'__inference_dense1_layer_call_fn_115526

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_1144902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Л
_
C__inference_flatten_layer_call_and_return_conditional_losses_114399

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ї
|
'__inference_dense2_layer_call_fn_115546

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_1145172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
с)
Ў
A__inference_norm1_layer_call_and_return_conditional_losses_115602

inputs
assignmovingavg_115577
assignmovingavg_1_115583 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Р*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Р2
moments/StopGradientЅ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indicesГ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Р*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Р*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Р*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/115577*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_115577*
_output_shapes	
:Р*
dtype02 
AssignMovingAvg/ReadVariableOpФ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/115577*
_output_shapes	
:Р2
AssignMovingAvg/subЛ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/115577*
_output_shapes	
:Р2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_115577AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/115577*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЄ
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/115583*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_115583*
_output_shapes	
:Р*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЮ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/115583*
_output_shapes	
:Р2
AssignMovingAvg_1/subХ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/115583*
_output_shapes	
:Р2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_115583AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/115583*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Р2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Р2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:Р2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Р2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Р2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
batchnorm/add_1Ж
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџР::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

	
A__inference_model_layer_call_and_return_conditional_losses_115300
input_1(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource 
norm1_assignmovingavg_115220"
norm1_assignmovingavg_1_115226&
"norm1_cast_readvariableop_resource(
$norm1_cast_1_readvariableop_resource 
norm2_assignmovingavg_115252"
norm2_assignmovingavg_1_115258&
"norm2_cast_readvariableop_resource(
$norm2_cast_1_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityЂ)norm1/AssignMovingAvg/AssignSubVariableOpЂ+norm1/AssignMovingAvg_1/AssignSubVariableOpЂ)norm2/AssignMovingAvg/AssignSubVariableOpЂ+norm2/AssignMovingAvg_1/AssignSubVariableOp_
ConstConst*
_output_shapes
:*
dtype0*
valueB"   [   2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЎ
splitSplitVinput_1Const:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':џџџџџџџџџ :џџџџџџџџџ[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
ReshapeЇ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpР
conv1/Conv2DConv2DReshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1/Conv2D
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp 
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

conv1/ReluЇ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOpШ
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2/Conv2D
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp 
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

conv2/ReluЏ
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
pool1/MaxPoolЇ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOpХ
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv3/Conv2D
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp 
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

conv3/ReluЇ
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv4/Conv2D/ReadVariableOpЧ
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv4/Conv2D
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOp 
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

conv4/ReluЏ
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
pool2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten/Const
flatten/ReshapeReshapepool2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/Reshape
$norm1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2&
$norm1/moments/mean/reduction_indicesД
norm1/moments/meanMeanflatten/Reshape:output:0-norm1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Р*
	keep_dims(2
norm1/moments/mean
norm1/moments/StopGradientStopGradientnorm1/moments/mean:output:0*
T0*
_output_shapes
:	Р2
norm1/moments/StopGradientЩ
norm1/moments/SquaredDifferenceSquaredDifferenceflatten/Reshape:output:0#norm1/moments/StopGradient:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2!
norm1/moments/SquaredDifference
(norm1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2*
(norm1/moments/variance/reduction_indicesЫ
norm1/moments/varianceMean#norm1/moments/SquaredDifference:z:01norm1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Р*
	keep_dims(2
norm1/moments/variance
norm1/moments/SqueezeSqueezenorm1/moments/mean:output:0*
T0*
_output_shapes	
:Р*
squeeze_dims
 2
norm1/moments/Squeeze
norm1/moments/Squeeze_1Squeezenorm1/moments/variance:output:0*
T0*
_output_shapes	
:Р*
squeeze_dims
 2
norm1/moments/Squeeze_1А
norm1/AssignMovingAvg/decayConst*/
_class%
#!loc:@norm1/AssignMovingAvg/115220*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
norm1/AssignMovingAvg/decayІ
$norm1/AssignMovingAvg/ReadVariableOpReadVariableOpnorm1_assignmovingavg_115220*
_output_shapes	
:Р*
dtype02&
$norm1/AssignMovingAvg/ReadVariableOpт
norm1/AssignMovingAvg/subSub,norm1/AssignMovingAvg/ReadVariableOp:value:0norm1/moments/Squeeze:output:0*
T0*/
_class%
#!loc:@norm1/AssignMovingAvg/115220*
_output_shapes	
:Р2
norm1/AssignMovingAvg/subй
norm1/AssignMovingAvg/mulMulnorm1/AssignMovingAvg/sub:z:0$norm1/AssignMovingAvg/decay:output:0*
T0*/
_class%
#!loc:@norm1/AssignMovingAvg/115220*
_output_shapes	
:Р2
norm1/AssignMovingAvg/mulЅ
)norm1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpnorm1_assignmovingavg_115220norm1/AssignMovingAvg/mul:z:0%^norm1/AssignMovingAvg/ReadVariableOp*/
_class%
#!loc:@norm1/AssignMovingAvg/115220*
_output_shapes
 *
dtype02+
)norm1/AssignMovingAvg/AssignSubVariableOpЖ
norm1/AssignMovingAvg_1/decayConst*1
_class'
%#loc:@norm1/AssignMovingAvg_1/115226*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
norm1/AssignMovingAvg_1/decayЌ
&norm1/AssignMovingAvg_1/ReadVariableOpReadVariableOpnorm1_assignmovingavg_1_115226*
_output_shapes	
:Р*
dtype02(
&norm1/AssignMovingAvg_1/ReadVariableOpь
norm1/AssignMovingAvg_1/subSub.norm1/AssignMovingAvg_1/ReadVariableOp:value:0 norm1/moments/Squeeze_1:output:0*
T0*1
_class'
%#loc:@norm1/AssignMovingAvg_1/115226*
_output_shapes	
:Р2
norm1/AssignMovingAvg_1/subу
norm1/AssignMovingAvg_1/mulMulnorm1/AssignMovingAvg_1/sub:z:0&norm1/AssignMovingAvg_1/decay:output:0*
T0*1
_class'
%#loc:@norm1/AssignMovingAvg_1/115226*
_output_shapes	
:Р2
norm1/AssignMovingAvg_1/mulБ
+norm1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpnorm1_assignmovingavg_1_115226norm1/AssignMovingAvg_1/mul:z:0'^norm1/AssignMovingAvg_1/ReadVariableOp*1
_class'
%#loc:@norm1/AssignMovingAvg_1/115226*
_output_shapes
 *
dtype02-
+norm1/AssignMovingAvg_1/AssignSubVariableOp
norm1/Cast/ReadVariableOpReadVariableOp"norm1_cast_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast/ReadVariableOp
norm1/Cast_1/ReadVariableOpReadVariableOp$norm1_cast_1_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast_1/ReadVariableOps
norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
norm1/batchnorm/add/y
norm1/batchnorm/addAddV2 norm1/moments/Squeeze_1:output:0norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/addv
norm1/batchnorm/RsqrtRsqrtnorm1/batchnorm/add:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/Rsqrt
norm1/batchnorm/mulMulnorm1/batchnorm/Rsqrt:y:0#norm1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/mul
norm1/batchnorm/mul_1Mulflatten/Reshape:output:0norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
norm1/batchnorm/mul_1
norm1/batchnorm/mul_2Mulnorm1/moments/Squeeze:output:0norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/mul_2
norm1/batchnorm/subSub!norm1/Cast/ReadVariableOp:value:0norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/sub
norm1/batchnorm/add_1AddV2norm1/batchnorm/mul_1:z:0norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
norm1/batchnorm/add_1
$norm2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2&
$norm2/moments/mean/reduction_indicesЉ
norm2/moments/meanMeansplit:output:1-norm2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
norm2/moments/mean
norm2/moments/StopGradientStopGradientnorm2/moments/mean:output:0*
T0*
_output_shapes

:[2
norm2/moments/StopGradientО
norm2/moments/SquaredDifferenceSquaredDifferencesplit:output:1#norm2/moments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџ[2!
norm2/moments/SquaredDifference
(norm2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2*
(norm2/moments/variance/reduction_indicesЪ
norm2/moments/varianceMean#norm2/moments/SquaredDifference:z:01norm2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
norm2/moments/variance
norm2/moments/SqueezeSqueezenorm2/moments/mean:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
norm2/moments/Squeeze
norm2/moments/Squeeze_1Squeezenorm2/moments/variance:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
norm2/moments/Squeeze_1А
norm2/AssignMovingAvg/decayConst*/
_class%
#!loc:@norm2/AssignMovingAvg/115252*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
norm2/AssignMovingAvg/decayЅ
$norm2/AssignMovingAvg/ReadVariableOpReadVariableOpnorm2_assignmovingavg_115252*
_output_shapes
:[*
dtype02&
$norm2/AssignMovingAvg/ReadVariableOpс
norm2/AssignMovingAvg/subSub,norm2/AssignMovingAvg/ReadVariableOp:value:0norm2/moments/Squeeze:output:0*
T0*/
_class%
#!loc:@norm2/AssignMovingAvg/115252*
_output_shapes
:[2
norm2/AssignMovingAvg/subи
norm2/AssignMovingAvg/mulMulnorm2/AssignMovingAvg/sub:z:0$norm2/AssignMovingAvg/decay:output:0*
T0*/
_class%
#!loc:@norm2/AssignMovingAvg/115252*
_output_shapes
:[2
norm2/AssignMovingAvg/mulЅ
)norm2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpnorm2_assignmovingavg_115252norm2/AssignMovingAvg/mul:z:0%^norm2/AssignMovingAvg/ReadVariableOp*/
_class%
#!loc:@norm2/AssignMovingAvg/115252*
_output_shapes
 *
dtype02+
)norm2/AssignMovingAvg/AssignSubVariableOpЖ
norm2/AssignMovingAvg_1/decayConst*1
_class'
%#loc:@norm2/AssignMovingAvg_1/115258*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
norm2/AssignMovingAvg_1/decayЋ
&norm2/AssignMovingAvg_1/ReadVariableOpReadVariableOpnorm2_assignmovingavg_1_115258*
_output_shapes
:[*
dtype02(
&norm2/AssignMovingAvg_1/ReadVariableOpы
norm2/AssignMovingAvg_1/subSub.norm2/AssignMovingAvg_1/ReadVariableOp:value:0 norm2/moments/Squeeze_1:output:0*
T0*1
_class'
%#loc:@norm2/AssignMovingAvg_1/115258*
_output_shapes
:[2
norm2/AssignMovingAvg_1/subт
norm2/AssignMovingAvg_1/mulMulnorm2/AssignMovingAvg_1/sub:z:0&norm2/AssignMovingAvg_1/decay:output:0*
T0*1
_class'
%#loc:@norm2/AssignMovingAvg_1/115258*
_output_shapes
:[2
norm2/AssignMovingAvg_1/mulБ
+norm2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpnorm2_assignmovingavg_1_115258norm2/AssignMovingAvg_1/mul:z:0'^norm2/AssignMovingAvg_1/ReadVariableOp*1
_class'
%#loc:@norm2/AssignMovingAvg_1/115258*
_output_shapes
 *
dtype02-
+norm2/AssignMovingAvg_1/AssignSubVariableOp
norm2/Cast/ReadVariableOpReadVariableOp"norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast/ReadVariableOp
norm2/Cast_1/ReadVariableOpReadVariableOp$norm2_cast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_1/ReadVariableOps
norm2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
norm2/batchnorm/add/y
norm2/batchnorm/addAddV2 norm2/moments/Squeeze_1:output:0norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
norm2/batchnorm/addu
norm2/batchnorm/RsqrtRsqrtnorm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/Rsqrt
norm2/batchnorm/mulMulnorm2/batchnorm/Rsqrt:y:0#norm2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul
norm2/batchnorm/mul_1Mulsplit:output:1norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
norm2/batchnorm/mul_1
norm2/batchnorm/mul_2Mulnorm2/moments/Squeeze:output:0norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul_2
norm2/batchnorm/subSub!norm2/Cast/ReadVariableOp:value:0norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/sub
norm2/batchnorm/add_1AddV2norm2/batchnorm/mul_1:z:0norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
norm2/batchnorm/add_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЄ
concatConcatV2norm1/batchnorm/add_1:z:0norm2/batchnorm/add_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ2
concatЄ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulconcat:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/MatMulЂ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/ReluЃ
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense2/MatMul/ReadVariableOp
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/MatMulЁ
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense2/BiasAdd/ReadVariableOp
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/ReluЂ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldense2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/ReluЁ
IdentityIdentityoutput/Relu:activations:0*^norm1/AssignMovingAvg/AssignSubVariableOp,^norm1/AssignMovingAvg_1/AssignSubVariableOp*^norm2/AssignMovingAvg/AssignSubVariableOp,^norm2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл ::::::::::::::::::::::2V
)norm1/AssignMovingAvg/AssignSubVariableOp)norm1/AssignMovingAvg/AssignSubVariableOp2Z
+norm1/AssignMovingAvg_1/AssignSubVariableOp+norm1/AssignMovingAvg_1/AssignSubVariableOp2V
)norm2/AssignMovingAvg/AssignSubVariableOp)norm2/AssignMovingAvg/AssignSubVariableOp2Z
+norm2/AssignMovingAvg_1/AssignSubVariableOp+norm2/AssignMovingAvg_1/AssignSubVariableOp:Q M
(
_output_shapes
:џџџџџџџџџл 
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ц
Њ
B__inference_dense2_layer_call_and_return_conditional_losses_114517

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
рb

A__inference_model_layer_call_and_return_conditional_losses_115073
x(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource&
"norm1_cast_readvariableop_resource(
$norm1_cast_1_readvariableop_resource(
$norm1_cast_2_readvariableop_resource(
$norm1_cast_3_readvariableop_resource&
"norm2_cast_readvariableop_resource(
$norm2_cast_1_readvariableop_resource(
$norm2_cast_2_readvariableop_resource(
$norm2_cast_3_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"   [   2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЈ
splitSplitVxConst:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':џџџџџџџџџ :џџџџџџџџџ[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
ReshapeЇ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpР
conv1/Conv2DConv2DReshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1/Conv2D
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp 
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

conv1/ReluЇ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOpШ
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2/Conv2D
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp 
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

conv2/ReluЏ
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
pool1/MaxPoolЇ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOpХ
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv3/Conv2D
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp 
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

conv3/ReluЇ
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv4/Conv2D/ReadVariableOpЧ
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv4/Conv2D
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOp 
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

conv4/ReluЏ
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
pool2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten/Const
flatten/ReshapeReshapepool2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/Reshape
norm1/Cast/ReadVariableOpReadVariableOp"norm1_cast_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast/ReadVariableOp
norm1/Cast_1/ReadVariableOpReadVariableOp$norm1_cast_1_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast_1/ReadVariableOp
norm1/Cast_2/ReadVariableOpReadVariableOp$norm1_cast_2_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast_2/ReadVariableOp
norm1/Cast_3/ReadVariableOpReadVariableOp$norm1_cast_3_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast_3/ReadVariableOps
norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
norm1/batchnorm/add/y
norm1/batchnorm/addAddV2#norm1/Cast_1/ReadVariableOp:value:0norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/addv
norm1/batchnorm/RsqrtRsqrtnorm1/batchnorm/add:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/Rsqrt
norm1/batchnorm/mulMulnorm1/batchnorm/Rsqrt:y:0#norm1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/mul
norm1/batchnorm/mul_1Mulflatten/Reshape:output:0norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
norm1/batchnorm/mul_1
norm1/batchnorm/mul_2Mul!norm1/Cast/ReadVariableOp:value:0norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/mul_2
norm1/batchnorm/subSub#norm1/Cast_2/ReadVariableOp:value:0norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/sub
norm1/batchnorm/add_1AddV2norm1/batchnorm/mul_1:z:0norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
norm1/batchnorm/add_1
norm2/Cast/ReadVariableOpReadVariableOp"norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast/ReadVariableOp
norm2/Cast_1/ReadVariableOpReadVariableOp$norm2_cast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_1/ReadVariableOp
norm2/Cast_2/ReadVariableOpReadVariableOp$norm2_cast_2_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_2/ReadVariableOp
norm2/Cast_3/ReadVariableOpReadVariableOp$norm2_cast_3_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_3/ReadVariableOps
norm2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
norm2/batchnorm/add/y
norm2/batchnorm/addAddV2#norm2/Cast_1/ReadVariableOp:value:0norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
norm2/batchnorm/addu
norm2/batchnorm/RsqrtRsqrtnorm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/Rsqrt
norm2/batchnorm/mulMulnorm2/batchnorm/Rsqrt:y:0#norm2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul
norm2/batchnorm/mul_1Mulsplit:output:1norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
norm2/batchnorm/mul_1
norm2/batchnorm/mul_2Mul!norm2/Cast/ReadVariableOp:value:0norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul_2
norm2/batchnorm/subSub#norm2/Cast_2/ReadVariableOp:value:0norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/sub
norm2/batchnorm/add_1AddV2norm2/batchnorm/mul_1:z:0norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
norm2/batchnorm/add_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЄ
concatConcatV2norm1/batchnorm/add_1:z:0norm2/batchnorm/add_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ2
concatЄ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulconcat:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/MatMulЂ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/ReluЃ
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense2/MatMul/ReadVariableOp
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/MatMulЁ
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense2/BiasAdd/ReadVariableOp
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/ReluЂ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldense2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/Relum
IdentityIdentityoutput/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл :::::::::::::::::::::::K G
(
_output_shapes
:џџџџџџџџџл 

_user_specified_nameX:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ь
Њ
B__inference_dense1_layer_call_and_return_conditional_losses_115517

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ѓ?

A__inference_model_layer_call_and_return_conditional_losses_114700
x
conv1_114641
conv1_114643
conv2_114646
conv2_114648
conv3_114652
conv3_114654
conv4_114657
conv4_114659
norm1_114664
norm1_114666
norm1_114668
norm1_114670
norm2_114673
norm2_114675
norm2_114677
norm2_114679
dense1_114684
dense1_114686
dense2_114689
dense2_114691
output_114694
output_114696
identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂconv3/StatefulPartitionedCallЂconv4/StatefulPartitionedCallЂdense1/StatefulPartitionedCallЂdense2/StatefulPartitionedCallЂnorm1/StatefulPartitionedCallЂnorm2/StatefulPartitionedCallЂoutput/StatefulPartitionedCall_
ConstConst*
_output_shapes
:*
dtype0*
valueB"   [   2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЈ
splitSplitVxConst:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':џџџџџџџџџ :џџџџџџџџџ[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
Reshapeј
conv1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1_114641conv1_114643*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1139812
conv1/StatefulPartitionedCall
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_114646conv2_114648*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1140032
conv2/StatefulPartitionedCallд
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_pool1_layer_call_and_return_conditional_losses_1140192
pool1/PartitionedCall
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv3_114652conv3_114654*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_1140372
conv3/StatefulPartitionedCall
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0conv4_114657conv4_114659*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_1140592
conv4/StatefulPartitionedCallд
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_pool2_layer_call_and_return_conditional_losses_1140752
pool2/PartitionedCallЫ
flatten/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1143992
flatten/PartitionedCallЁ
norm1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0norm1_114664norm1_114666norm1_114668norm1_114670*
Tin	
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР*&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_norm1_layer_call_and_return_conditional_losses_1142102
norm1/StatefulPartitionedCall
norm2/StatefulPartitionedCallStatefulPartitionedCallsplit:output:1norm2_114673norm2_114675norm2_114677norm2_114679*
Tin	
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ[*&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_norm2_layer_call_and_return_conditional_losses_1143502
norm2/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisО
concatConcatV2&norm1/StatefulPartitionedCall:output:0&norm2/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ2
concatѕ
dense1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense1_114684dense1_114686*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_1144902 
dense1/StatefulPartitionedCall
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_114689dense2_114691*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_1145172 
dense2/StatefulPartitionedCall
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_114694output_114696*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1145442 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^norm1/StatefulPartitionedCall^norm2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл ::::::::::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2>
norm1/StatefulPartitionedCallnorm1/StatefulPartitionedCall2>
norm2/StatefulPartitionedCallnorm2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:K G
(
_output_shapes
:џџџџџџџџџл 

_user_specified_nameX:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
е
­
&__inference_model_layer_call_fn_115122
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1147002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:џџџџџџџџџл 

_user_specified_nameX:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
сn
ъ
!__inference__wrapped_model_113968
input_1.
*model_conv1_conv2d_readvariableop_resource/
+model_conv1_biasadd_readvariableop_resource.
*model_conv2_conv2d_readvariableop_resource/
+model_conv2_biasadd_readvariableop_resource.
*model_conv3_conv2d_readvariableop_resource/
+model_conv3_biasadd_readvariableop_resource.
*model_conv4_conv2d_readvariableop_resource/
+model_conv4_biasadd_readvariableop_resource,
(model_norm1_cast_readvariableop_resource.
*model_norm1_cast_1_readvariableop_resource.
*model_norm1_cast_2_readvariableop_resource.
*model_norm1_cast_3_readvariableop_resource,
(model_norm2_cast_readvariableop_resource.
*model_norm2_cast_1_readvariableop_resource.
*model_norm2_cast_2_readvariableop_resource.
*model_norm2_cast_3_readvariableop_resource/
+model_dense1_matmul_readvariableop_resource0
,model_dense1_biasadd_readvariableop_resource/
+model_dense2_matmul_readvariableop_resource0
,model_dense2_biasadd_readvariableop_resource/
+model_output_matmul_readvariableop_resource0
,model_output_biasadd_readvariableop_resource
identityk
model/ConstConst*
_output_shapes
:*
dtype0*
valueB"   [   2
model/Constp
model/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
model/split/split_dimЦ
model/splitSplitVinput_1model/Const:output:0model/split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':џџџџџџџџџ :џџџџџџџџџ[*
	num_split2
model/split
model/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
model/Reshape/shape
model/ReshapeReshapemodel/split:output:0model/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
model/ReshapeЙ
!model/conv1/Conv2D/ReadVariableOpReadVariableOp*model_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/conv1/Conv2D/ReadVariableOpи
model/conv1/Conv2DConv2Dmodel/Reshape:output:0)model/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
model/conv1/Conv2DА
"model/conv1/BiasAdd/ReadVariableOpReadVariableOp+model_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/conv1/BiasAdd/ReadVariableOpИ
model/conv1/BiasAddBiasAddmodel/conv1/Conv2D:output:0*model/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
model/conv1/BiasAdd
model/conv1/ReluRelumodel/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
model/conv1/ReluЙ
!model/conv2/Conv2D/ReadVariableOpReadVariableOp*model_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!model/conv2/Conv2D/ReadVariableOpр
model/conv2/Conv2DConv2Dmodel/conv1/Relu:activations:0)model/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
model/conv2/Conv2DА
"model/conv2/BiasAdd/ReadVariableOpReadVariableOp+model_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/conv2/BiasAdd/ReadVariableOpИ
model/conv2/BiasAddBiasAddmodel/conv2/Conv2D:output:0*model/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
model/conv2/BiasAdd
model/conv2/ReluRelumodel/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
model/conv2/ReluС
model/pool1/MaxPoolMaxPoolmodel/conv2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
model/pool1/MaxPoolЙ
!model/conv3/Conv2D/ReadVariableOpReadVariableOp*model_conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!model/conv3/Conv2D/ReadVariableOpн
model/conv3/Conv2DConv2Dmodel/pool1/MaxPool:output:0)model/conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
model/conv3/Conv2DА
"model/conv3/BiasAdd/ReadVariableOpReadVariableOp+model_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/conv3/BiasAdd/ReadVariableOpИ
model/conv3/BiasAddBiasAddmodel/conv3/Conv2D:output:0*model/conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
model/conv3/BiasAdd
model/conv3/ReluRelumodel/conv3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
model/conv3/ReluЙ
!model/conv4/Conv2D/ReadVariableOpReadVariableOp*model_conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02#
!model/conv4/Conv2D/ReadVariableOpп
model/conv4/Conv2DConv2Dmodel/conv3/Relu:activations:0)model/conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
model/conv4/Conv2DА
"model/conv4/BiasAdd/ReadVariableOpReadVariableOp+model_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/conv4/BiasAdd/ReadVariableOpИ
model/conv4/BiasAddBiasAddmodel/conv4/Conv2D:output:0*model/conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
model/conv4/BiasAdd
model/conv4/ReluRelumodel/conv4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
model/conv4/ReluС
model/pool2/MaxPoolMaxPoolmodel/conv4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
model/pool2/MaxPool{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
model/flatten/ConstЈ
model/flatten/ReshapeReshapemodel/pool2/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
model/flatten/ReshapeЈ
model/norm1/Cast/ReadVariableOpReadVariableOp(model_norm1_cast_readvariableop_resource*
_output_shapes	
:Р*
dtype02!
model/norm1/Cast/ReadVariableOpЎ
!model/norm1/Cast_1/ReadVariableOpReadVariableOp*model_norm1_cast_1_readvariableop_resource*
_output_shapes	
:Р*
dtype02#
!model/norm1/Cast_1/ReadVariableOpЎ
!model/norm1/Cast_2/ReadVariableOpReadVariableOp*model_norm1_cast_2_readvariableop_resource*
_output_shapes	
:Р*
dtype02#
!model/norm1/Cast_2/ReadVariableOpЎ
!model/norm1/Cast_3/ReadVariableOpReadVariableOp*model_norm1_cast_3_readvariableop_resource*
_output_shapes	
:Р*
dtype02#
!model/norm1/Cast_3/ReadVariableOp
model/norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
model/norm1/batchnorm/add/yЖ
model/norm1/batchnorm/addAddV2)model/norm1/Cast_1/ReadVariableOp:value:0$model/norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Р2
model/norm1/batchnorm/add
model/norm1/batchnorm/RsqrtRsqrtmodel/norm1/batchnorm/add:z:0*
T0*
_output_shapes	
:Р2
model/norm1/batchnorm/RsqrtЏ
model/norm1/batchnorm/mulMulmodel/norm1/batchnorm/Rsqrt:y:0)model/norm1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:Р2
model/norm1/batchnorm/mulГ
model/norm1/batchnorm/mul_1Mulmodel/flatten/Reshape:output:0model/norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
model/norm1/batchnorm/mul_1Џ
model/norm1/batchnorm/mul_2Mul'model/norm1/Cast/ReadVariableOp:value:0model/norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Р2
model/norm1/batchnorm/mul_2Џ
model/norm1/batchnorm/subSub)model/norm1/Cast_2/ReadVariableOp:value:0model/norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Р2
model/norm1/batchnorm/subЖ
model/norm1/batchnorm/add_1AddV2model/norm1/batchnorm/mul_1:z:0model/norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
model/norm1/batchnorm/add_1Ї
model/norm2/Cast/ReadVariableOpReadVariableOp(model_norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02!
model/norm2/Cast/ReadVariableOp­
!model/norm2/Cast_1/ReadVariableOpReadVariableOp*model_norm2_cast_1_readvariableop_resource*
_output_shapes
:[*
dtype02#
!model/norm2/Cast_1/ReadVariableOp­
!model/norm2/Cast_2/ReadVariableOpReadVariableOp*model_norm2_cast_2_readvariableop_resource*
_output_shapes
:[*
dtype02#
!model/norm2/Cast_2/ReadVariableOp­
!model/norm2/Cast_3/ReadVariableOpReadVariableOp*model_norm2_cast_3_readvariableop_resource*
_output_shapes
:[*
dtype02#
!model/norm2/Cast_3/ReadVariableOp
model/norm2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
model/norm2/batchnorm/add/yЕ
model/norm2/batchnorm/addAddV2)model/norm2/Cast_1/ReadVariableOp:value:0$model/norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/add
model/norm2/batchnorm/RsqrtRsqrtmodel/norm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/RsqrtЎ
model/norm2/batchnorm/mulMulmodel/norm2/batchnorm/Rsqrt:y:0)model/norm2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/mulЈ
model/norm2/batchnorm/mul_1Mulmodel/split:output:1model/norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
model/norm2/batchnorm/mul_1Ў
model/norm2/batchnorm/mul_2Mul'model/norm2/Cast/ReadVariableOp:value:0model/norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/mul_2Ў
model/norm2/batchnorm/subSub)model/norm2/Cast_2/ReadVariableOp:value:0model/norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/subЕ
model/norm2/batchnorm/add_1AddV2model/norm2/batchnorm/mul_1:z:0model/norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
model/norm2/batchnorm/add_1h
model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concat/axisТ
model/concatConcatV2model/norm1/batchnorm/add_1:z:0model/norm2/batchnorm/add_1:z:0model/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ2
model/concatЖ
"model/dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02$
"model/dense1/MatMul/ReadVariableOpЊ
model/dense1/MatMulMatMulmodel/concat:output:0*model/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/dense1/MatMulД
#model/dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#model/dense1/BiasAdd/ReadVariableOpЖ
model/dense1/BiasAddBiasAddmodel/dense1/MatMul:product:0+model/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/dense1/BiasAdd
model/dense1/ReluRelumodel/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/dense1/ReluЕ
"model/dense2/MatMul/ReadVariableOpReadVariableOp+model_dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02$
"model/dense2/MatMul/ReadVariableOpГ
model/dense2/MatMulMatMulmodel/dense1/Relu:activations:0*model/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model/dense2/MatMulГ
#model/dense2/BiasAdd/ReadVariableOpReadVariableOp,model_dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/dense2/BiasAdd/ReadVariableOpЕ
model/dense2/BiasAddBiasAddmodel/dense2/MatMul:product:0+model/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model/dense2/BiasAdd
model/dense2/ReluRelumodel/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model/dense2/ReluД
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02$
"model/output/MatMul/ReadVariableOpГ
model/output/MatMulMatMulmodel/dense2/Relu:activations:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/output/MatMulГ
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/output/BiasAdd/ReadVariableOpЕ
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/output/BiasAdd
model/output/ReluRelumodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/output/Relus
IdentityIdentitymodel/output/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл :::::::::::::::::::::::Q M
(
_output_shapes
:џџџџџџџџџл 
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
е
­
&__inference_model_layer_call_fn_115171
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1147002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:џџџџџџџџџл 

_user_specified_nameX:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
і
]
A__inference_pool2_layer_call_and_return_conditional_losses_114075

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Л
_
C__inference_flatten_layer_call_and_return_conditional_losses_115501

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
у
Њ
B__inference_output_layer_call_and_return_conditional_losses_115557

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
л
{
&__inference_conv1_layer_call_fn_113991

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_1139812
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
с)
Ў
A__inference_norm1_layer_call_and_return_conditional_losses_114177

inputs
assignmovingavg_114152
assignmovingavg_1_114158 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityЂ#AssignMovingAvg/AssignSubVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Р*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Р2
moments/StopGradientЅ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indicesГ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Р*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Р*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Р*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/114152*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_114152*
_output_shapes	
:Р*
dtype02 
AssignMovingAvg/ReadVariableOpФ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/114152*
_output_shapes	
:Р2
AssignMovingAvg/subЛ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/114152*
_output_shapes	
:Р2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_114152AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/114152*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpЄ
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/114158*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_114158*
_output_shapes	
:Р*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЮ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/114158*
_output_shapes	
:Р2
AssignMovingAvg_1/subХ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/114158*
_output_shapes	
:Р2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_114158AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/114158*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Р2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Р2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:Р2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Р2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Р2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
batchnorm/add_1Ж
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџР::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ю
A__inference_norm1_layer_call_and_return_conditional_losses_114210

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:Р*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Р2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Р2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:Р2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Р2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Р2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџР:::::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
А

Љ
A__inference_conv1_layer_call_and_return_conditional_losses_113981

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
њ
B
&__inference_pool2_layer_call_fn_114081

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_pool2_layer_call_and_return_conditional_losses_1140752
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Х
Б
$__inference_signature_wrapper_114847
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__wrapped_model_1139682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџл 
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ю

&__inference_norm2_layer_call_fn_115717

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ[*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_norm2_layer_call_and_return_conditional_losses_1143172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ[::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ[
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ћ`

"__inference__traced_restore_115901
file_prefix'
#assignvariableop_model_conv1_kernel'
#assignvariableop_1_model_conv1_bias)
%assignvariableop_2_model_conv2_kernel'
#assignvariableop_3_model_conv2_bias)
%assignvariableop_4_model_conv3_kernel'
#assignvariableop_5_model_conv3_bias)
%assignvariableop_6_model_conv4_kernel'
#assignvariableop_7_model_conv4_bias*
&assignvariableop_8_model_dense1_kernel(
$assignvariableop_9_model_dense1_bias+
'assignvariableop_10_model_dense2_kernel)
%assignvariableop_11_model_dense2_bias+
'assignvariableop_12_model_output_kernel)
%assignvariableop_13_model_output_bias)
%assignvariableop_14_model_norm1_gamma(
$assignvariableop_15_model_norm1_beta/
+assignvariableop_16_model_norm1_moving_mean3
/assignvariableop_17_model_norm1_moving_variance)
%assignvariableop_18_model_norm2_gamma(
$assignvariableop_19_model_norm2_beta/
+assignvariableop_20_model_norm2_moving_mean3
/assignvariableop_21_model_norm2_moving_variance
identity_23ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Љ
valueBB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d3/bias/.ATTRIBUTES/VARIABLE_VALUEB)b_norm_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB(b_norm_1/beta/.ATTRIBUTES/VARIABLE_VALUEB/b_norm_1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3b_norm_1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)b_norm_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB(b_norm_2/beta/.ATTRIBUTES/VARIABLE_VALUEB/b_norm_2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3b_norm_2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesК
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp#assignvariableop_model_conv1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp#assignvariableop_1_model_conv1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp%assignvariableop_2_model_conv2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp#assignvariableop_3_model_conv2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp%assignvariableop_4_model_conv3_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp#assignvariableop_5_model_conv3_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp%assignvariableop_6_model_conv4_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp#assignvariableop_7_model_conv4_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp&assignvariableop_8_model_dense1_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp$assignvariableop_9_model_dense1_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10 
AssignVariableOp_10AssignVariableOp'assignvariableop_10_model_dense2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOp%assignvariableop_11_model_dense2_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12 
AssignVariableOp_12AssignVariableOp'assignvariableop_12_model_output_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOp%assignvariableop_13_model_output_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOp%assignvariableop_14_model_norm1_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOp$assignvariableop_15_model_norm1_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Є
AssignVariableOp_16AssignVariableOp+assignvariableop_16_model_norm1_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Ј
AssignVariableOp_17AssignVariableOp/assignvariableop_17_model_norm1_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18
AssignVariableOp_18AssignVariableOp%assignvariableop_18_model_norm2_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOp$assignvariableop_19_model_norm2_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Є
AssignVariableOp_20AssignVariableOp+assignvariableop_20_model_norm2_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Ј
AssignVariableOp_21AssignVariableOp/assignvariableop_21_model_norm2_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpТ
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22Я
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ц
Њ
B__inference_dense2_layer_call_and_return_conditional_losses_115537

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Џ

Љ
A__inference_conv4_layer_call_and_return_conditional_losses_114059

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
л
{
&__inference_conv2_layer_call_fn_114013

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_1140032
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ѕ
|
'__inference_output_layer_call_fn_115566

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1145442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
љ
	
A__inference_model_layer_call_and_return_conditional_losses_114976
x(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource 
norm1_assignmovingavg_114896"
norm1_assignmovingavg_1_114902&
"norm1_cast_readvariableop_resource(
$norm1_cast_1_readvariableop_resource 
norm2_assignmovingavg_114928"
norm2_assignmovingavg_1_114934&
"norm2_cast_readvariableop_resource(
$norm2_cast_1_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityЂ)norm1/AssignMovingAvg/AssignSubVariableOpЂ+norm1/AssignMovingAvg_1/AssignSubVariableOpЂ)norm2/AssignMovingAvg/AssignSubVariableOpЂ+norm2/AssignMovingAvg_1/AssignSubVariableOp_
ConstConst*
_output_shapes
:*
dtype0*
valueB"   [   2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЈ
splitSplitVxConst:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':џџџџџџџџџ :џџџџџџџџџ[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
ReshapeЇ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpР
conv1/Conv2DConv2DReshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv1/Conv2D
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp 
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

conv1/ReluЇ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOpШ
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv2/Conv2D
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp 
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

conv2/ReluЏ
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
pool1/MaxPoolЇ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOpХ
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv3/Conv2D
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp 
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

conv3/ReluЇ
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv4/Conv2D/ReadVariableOpЧ
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv4/Conv2D
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOp 
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2

conv4/ReluЏ
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
pool2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten/Const
flatten/ReshapeReshapepool2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/Reshape
$norm1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2&
$norm1/moments/mean/reduction_indicesД
norm1/moments/meanMeanflatten/Reshape:output:0-norm1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Р*
	keep_dims(2
norm1/moments/mean
norm1/moments/StopGradientStopGradientnorm1/moments/mean:output:0*
T0*
_output_shapes
:	Р2
norm1/moments/StopGradientЩ
norm1/moments/SquaredDifferenceSquaredDifferenceflatten/Reshape:output:0#norm1/moments/StopGradient:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2!
norm1/moments/SquaredDifference
(norm1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2*
(norm1/moments/variance/reduction_indicesЫ
norm1/moments/varianceMean#norm1/moments/SquaredDifference:z:01norm1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Р*
	keep_dims(2
norm1/moments/variance
norm1/moments/SqueezeSqueezenorm1/moments/mean:output:0*
T0*
_output_shapes	
:Р*
squeeze_dims
 2
norm1/moments/Squeeze
norm1/moments/Squeeze_1Squeezenorm1/moments/variance:output:0*
T0*
_output_shapes	
:Р*
squeeze_dims
 2
norm1/moments/Squeeze_1А
norm1/AssignMovingAvg/decayConst*/
_class%
#!loc:@norm1/AssignMovingAvg/114896*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
norm1/AssignMovingAvg/decayІ
$norm1/AssignMovingAvg/ReadVariableOpReadVariableOpnorm1_assignmovingavg_114896*
_output_shapes	
:Р*
dtype02&
$norm1/AssignMovingAvg/ReadVariableOpт
norm1/AssignMovingAvg/subSub,norm1/AssignMovingAvg/ReadVariableOp:value:0norm1/moments/Squeeze:output:0*
T0*/
_class%
#!loc:@norm1/AssignMovingAvg/114896*
_output_shapes	
:Р2
norm1/AssignMovingAvg/subй
norm1/AssignMovingAvg/mulMulnorm1/AssignMovingAvg/sub:z:0$norm1/AssignMovingAvg/decay:output:0*
T0*/
_class%
#!loc:@norm1/AssignMovingAvg/114896*
_output_shapes	
:Р2
norm1/AssignMovingAvg/mulЅ
)norm1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpnorm1_assignmovingavg_114896norm1/AssignMovingAvg/mul:z:0%^norm1/AssignMovingAvg/ReadVariableOp*/
_class%
#!loc:@norm1/AssignMovingAvg/114896*
_output_shapes
 *
dtype02+
)norm1/AssignMovingAvg/AssignSubVariableOpЖ
norm1/AssignMovingAvg_1/decayConst*1
_class'
%#loc:@norm1/AssignMovingAvg_1/114902*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
norm1/AssignMovingAvg_1/decayЌ
&norm1/AssignMovingAvg_1/ReadVariableOpReadVariableOpnorm1_assignmovingavg_1_114902*
_output_shapes	
:Р*
dtype02(
&norm1/AssignMovingAvg_1/ReadVariableOpь
norm1/AssignMovingAvg_1/subSub.norm1/AssignMovingAvg_1/ReadVariableOp:value:0 norm1/moments/Squeeze_1:output:0*
T0*1
_class'
%#loc:@norm1/AssignMovingAvg_1/114902*
_output_shapes	
:Р2
norm1/AssignMovingAvg_1/subу
norm1/AssignMovingAvg_1/mulMulnorm1/AssignMovingAvg_1/sub:z:0&norm1/AssignMovingAvg_1/decay:output:0*
T0*1
_class'
%#loc:@norm1/AssignMovingAvg_1/114902*
_output_shapes	
:Р2
norm1/AssignMovingAvg_1/mulБ
+norm1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpnorm1_assignmovingavg_1_114902norm1/AssignMovingAvg_1/mul:z:0'^norm1/AssignMovingAvg_1/ReadVariableOp*1
_class'
%#loc:@norm1/AssignMovingAvg_1/114902*
_output_shapes
 *
dtype02-
+norm1/AssignMovingAvg_1/AssignSubVariableOp
norm1/Cast/ReadVariableOpReadVariableOp"norm1_cast_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast/ReadVariableOp
norm1/Cast_1/ReadVariableOpReadVariableOp$norm1_cast_1_readvariableop_resource*
_output_shapes	
:Р*
dtype02
norm1/Cast_1/ReadVariableOps
norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
norm1/batchnorm/add/y
norm1/batchnorm/addAddV2 norm1/moments/Squeeze_1:output:0norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/addv
norm1/batchnorm/RsqrtRsqrtnorm1/batchnorm/add:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/Rsqrt
norm1/batchnorm/mulMulnorm1/batchnorm/Rsqrt:y:0#norm1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/mul
norm1/batchnorm/mul_1Mulflatten/Reshape:output:0norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
norm1/batchnorm/mul_1
norm1/batchnorm/mul_2Mulnorm1/moments/Squeeze:output:0norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/mul_2
norm1/batchnorm/subSub!norm1/Cast/ReadVariableOp:value:0norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Р2
norm1/batchnorm/sub
norm1/batchnorm/add_1AddV2norm1/batchnorm/mul_1:z:0norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџР2
norm1/batchnorm/add_1
$norm2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2&
$norm2/moments/mean/reduction_indicesЉ
norm2/moments/meanMeansplit:output:1-norm2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
norm2/moments/mean
norm2/moments/StopGradientStopGradientnorm2/moments/mean:output:0*
T0*
_output_shapes

:[2
norm2/moments/StopGradientО
norm2/moments/SquaredDifferenceSquaredDifferencesplit:output:1#norm2/moments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџ[2!
norm2/moments/SquaredDifference
(norm2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2*
(norm2/moments/variance/reduction_indicesЪ
norm2/moments/varianceMean#norm2/moments/SquaredDifference:z:01norm2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
norm2/moments/variance
norm2/moments/SqueezeSqueezenorm2/moments/mean:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
norm2/moments/Squeeze
norm2/moments/Squeeze_1Squeezenorm2/moments/variance:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
norm2/moments/Squeeze_1А
norm2/AssignMovingAvg/decayConst*/
_class%
#!loc:@norm2/AssignMovingAvg/114928*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
norm2/AssignMovingAvg/decayЅ
$norm2/AssignMovingAvg/ReadVariableOpReadVariableOpnorm2_assignmovingavg_114928*
_output_shapes
:[*
dtype02&
$norm2/AssignMovingAvg/ReadVariableOpс
norm2/AssignMovingAvg/subSub,norm2/AssignMovingAvg/ReadVariableOp:value:0norm2/moments/Squeeze:output:0*
T0*/
_class%
#!loc:@norm2/AssignMovingAvg/114928*
_output_shapes
:[2
norm2/AssignMovingAvg/subи
norm2/AssignMovingAvg/mulMulnorm2/AssignMovingAvg/sub:z:0$norm2/AssignMovingAvg/decay:output:0*
T0*/
_class%
#!loc:@norm2/AssignMovingAvg/114928*
_output_shapes
:[2
norm2/AssignMovingAvg/mulЅ
)norm2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpnorm2_assignmovingavg_114928norm2/AssignMovingAvg/mul:z:0%^norm2/AssignMovingAvg/ReadVariableOp*/
_class%
#!loc:@norm2/AssignMovingAvg/114928*
_output_shapes
 *
dtype02+
)norm2/AssignMovingAvg/AssignSubVariableOpЖ
norm2/AssignMovingAvg_1/decayConst*1
_class'
%#loc:@norm2/AssignMovingAvg_1/114934*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
norm2/AssignMovingAvg_1/decayЋ
&norm2/AssignMovingAvg_1/ReadVariableOpReadVariableOpnorm2_assignmovingavg_1_114934*
_output_shapes
:[*
dtype02(
&norm2/AssignMovingAvg_1/ReadVariableOpы
norm2/AssignMovingAvg_1/subSub.norm2/AssignMovingAvg_1/ReadVariableOp:value:0 norm2/moments/Squeeze_1:output:0*
T0*1
_class'
%#loc:@norm2/AssignMovingAvg_1/114934*
_output_shapes
:[2
norm2/AssignMovingAvg_1/subт
norm2/AssignMovingAvg_1/mulMulnorm2/AssignMovingAvg_1/sub:z:0&norm2/AssignMovingAvg_1/decay:output:0*
T0*1
_class'
%#loc:@norm2/AssignMovingAvg_1/114934*
_output_shapes
:[2
norm2/AssignMovingAvg_1/mulБ
+norm2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpnorm2_assignmovingavg_1_114934norm2/AssignMovingAvg_1/mul:z:0'^norm2/AssignMovingAvg_1/ReadVariableOp*1
_class'
%#loc:@norm2/AssignMovingAvg_1/114934*
_output_shapes
 *
dtype02-
+norm2/AssignMovingAvg_1/AssignSubVariableOp
norm2/Cast/ReadVariableOpReadVariableOp"norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast/ReadVariableOp
norm2/Cast_1/ReadVariableOpReadVariableOp$norm2_cast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_1/ReadVariableOps
norm2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
norm2/batchnorm/add/y
norm2/batchnorm/addAddV2 norm2/moments/Squeeze_1:output:0norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
norm2/batchnorm/addu
norm2/batchnorm/RsqrtRsqrtnorm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/Rsqrt
norm2/batchnorm/mulMulnorm2/batchnorm/Rsqrt:y:0#norm2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul
norm2/batchnorm/mul_1Mulsplit:output:1norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
norm2/batchnorm/mul_1
norm2/batchnorm/mul_2Mulnorm2/moments/Squeeze:output:0norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul_2
norm2/batchnorm/subSub!norm2/Cast/ReadVariableOp:value:0norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/sub
norm2/batchnorm/add_1AddV2norm2/batchnorm/mul_1:z:0norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ[2
norm2/batchnorm/add_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЄ
concatConcatV2norm1/batchnorm/add_1:z:0norm2/batchnorm/add_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ2
concatЄ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulconcat:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/MatMulЂ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense1/ReluЃ
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense2/MatMul/ReadVariableOp
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/MatMulЁ
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense2/BiasAdd/ReadVariableOp
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense2/ReluЂ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldense2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/ReluЁ
IdentityIdentityoutput/Relu:activations:0*^norm1/AssignMovingAvg/AssignSubVariableOp,^norm1/AssignMovingAvg_1/AssignSubVariableOp*^norm2/AssignMovingAvg/AssignSubVariableOp,^norm2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл ::::::::::::::::::::::2V
)norm1/AssignMovingAvg/AssignSubVariableOp)norm1/AssignMovingAvg/AssignSubVariableOp2Z
+norm1/AssignMovingAvg_1/AssignSubVariableOp+norm1/AssignMovingAvg_1/AssignSubVariableOp2V
)norm2/AssignMovingAvg/AssignSubVariableOp)norm2/AssignMovingAvg/AssignSubVariableOp2Z
+norm2/AssignMovingAvg_1/AssignSubVariableOp+norm2/AssignMovingAvg_1/AssignSubVariableOp:K G
(
_output_shapes
:џџџџџџџџџл 

_user_specified_nameX:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
у
Њ
B__inference_output_layer_call_and_return_conditional_losses_114544

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ч
Г
&__inference_model_layer_call_fn_115446
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1147002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapesn
l:џџџџџџџџџл ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџл 
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
і
]
A__inference_pool1_layer_call_and_return_conditional_losses_114019

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ќ
serving_default
<
input_11
serving_default_input_1:0џџџџџџџџџл <
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Ь

	conv1
	conv2
	pool1
	conv3
	conv4
	pool2
flatten
d1
	d2

d3
b_norm_1
b_norm_2
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"б
_tf_keras_modelЗ{"class_name": "model", "name": "model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "model"}}
Н	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerќ{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 16, 16, 16]}}
Н	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+ &call_and_return_all_conditional_losses
Ё__call__"
_tf_keras_layerќ{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 14, 14, 16]}}
Щ
regularization_losses
	variables
 trainable_variables
!	keras_api
+Ђ&call_and_return_all_conditional_losses
Ѓ__call__"И
_tf_keras_layer{"class_name": "MaxPooling2D", "name": "pool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
К	

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+Є&call_and_return_all_conditional_losses
Ѕ__call__"
_tf_keras_layerљ{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 6, 6, 32]}}
К	

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+І&call_and_return_all_conditional_losses
Ї__call__"
_tf_keras_layerљ{"class_name": "Conv2D", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 6, 6, 32]}}
Щ
.regularization_losses
/	variables
0trainable_variables
1	keras_api
+Ј&call_and_return_all_conditional_losses
Љ__call__"И
_tf_keras_layer{"class_name": "MaxPooling2D", "name": "pool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
С
2regularization_losses
3	variables
4trainable_variables
5	keras_api
+Њ&call_and_return_all_conditional_losses
Ћ__call__"А
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ю

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
+Ќ&call_and_return_all_conditional_losses
­__call__"Ї
_tf_keras_layer{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 667}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 667]}}
Э

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+Ў&call_and_return_all_conditional_losses
Џ__call__"І
_tf_keras_layer{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 128]}}
Ъ

Bkernel
Cbias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"Ѓ
_tf_keras_layer{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 64]}}
ё
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"
_tf_keras_layer{"class_name": "BatchNormalization", "name": "norm1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "norm1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 576}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 576]}}
я
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"
_tf_keras_layerџ{"class_name": "BatchNormalization", "name": "norm2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "norm2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 91}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 91]}}
 "
trackable_list_wrapper
Ц
0
1
2
3
"4
#5
(6
)7
68
79
<10
=11
B12
C13
I14
J15
K16
L17
R18
S19
T20
U21"
trackable_list_wrapper
І
0
1
2
3
"4
#5
(6
)7
68
79
<10
=11
B12
C13
I14
J15
R16
S17"
trackable_list_wrapper
Ю
Znon_trainable_variables
[layer_metrics

\layers
regularization_losses
	variables
trainable_variables
]metrics
^layer_regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
Жserving_default"
signature_map
,:*2model/conv1/kernel
:2model/conv1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
_non_trainable_variables
`layer_metrics

alayers
regularization_losses
	variables
trainable_variables
bmetrics
clayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:* 2model/conv2/kernel
: 2model/conv2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
dnon_trainable_variables
elayer_metrics

flayers
regularization_losses
	variables
trainable_variables
gmetrics
hlayer_regularization_losses
Ё__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
inon_trainable_variables
jlayer_metrics

klayers
regularization_losses
	variables
 trainable_variables
lmetrics
mlayer_regularization_losses
Ѓ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
,:*  2model/conv3/kernel
: 2model/conv3/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
А
nnon_trainable_variables
olayer_metrics

players
$regularization_losses
%	variables
&trainable_variables
qmetrics
rlayer_regularization_losses
Ѕ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
,:* @2model/conv4/kernel
:@2model/conv4/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
А
snon_trainable_variables
tlayer_metrics

ulayers
*regularization_losses
+	variables
,trainable_variables
vmetrics
wlayer_regularization_losses
Ї__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
xnon_trainable_variables
ylayer_metrics

zlayers
.regularization_losses
/	variables
0trainable_variables
{metrics
|layer_regularization_losses
Љ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
}non_trainable_variables
~layer_metrics

layers
2regularization_losses
3	variables
4trainable_variables
metrics
 layer_regularization_losses
Ћ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
':%
2model/dense1/kernel
 :2model/dense1/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
Е
non_trainable_variables
layer_metrics
layers
8regularization_losses
9	variables
:trainable_variables
metrics
 layer_regularization_losses
­__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
&:$	@2model/dense2/kernel
:@2model/dense2/bias
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
Е
non_trainable_variables
layer_metrics
layers
>regularization_losses
?	variables
@trainable_variables
metrics
 layer_regularization_losses
Џ__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
%:#@2model/output/kernel
:2model/output/bias
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
Е
non_trainable_variables
layer_metrics
layers
Dregularization_losses
E	variables
Ftrainable_variables
metrics
 layer_regularization_losses
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :Р2model/norm1/gamma
:Р2model/norm1/beta
(:&Р (2model/norm1/moving_mean
,:*Р (2model/norm1/moving_variance
 "
trackable_list_wrapper
<
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
Е
non_trainable_variables
layer_metrics
layers
Mregularization_losses
N	variables
Otrainable_variables
metrics
 layer_regularization_losses
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:[2model/norm2/gamma
:[2model/norm2/beta
':%[ (2model/norm2/moving_mean
+:)[ (2model/norm2/moving_variance
 "
trackable_list_wrapper
<
R0
S1
T2
U3"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
Е
non_trainable_variables
layer_metrics
layers
Vregularization_losses
W	variables
Xtrainable_variables
metrics
 layer_regularization_losses
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
<
K0
L1
T2
U3"
trackable_list_wrapper
 "
trackable_dict_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Р2Н
A__inference_model_layer_call_and_return_conditional_losses_115073
A__inference_model_layer_call_and_return_conditional_losses_115300
A__inference_model_layer_call_and_return_conditional_losses_115397
A__inference_model_layer_call_and_return_conditional_losses_114976Ў
ЅВЁ
FullArgSpec$
args
jself
jX

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
р2н
!__inference__wrapped_model_113968З
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *'Ђ$
"
input_1џџџџџџџџџл 
д2б
&__inference_model_layer_call_fn_115495
&__inference_model_layer_call_fn_115446
&__inference_model_layer_call_fn_115122
&__inference_model_layer_call_fn_115171Ў
ЅВЁ
FullArgSpec$
args
jself
jX

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 2
A__inference_conv1_layer_call_and_return_conditional_losses_113981з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_conv1_layer_call_fn_113991з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 2
A__inference_conv2_layer_call_and_return_conditional_losses_114003з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_conv2_layer_call_fn_114013з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Љ2І
A__inference_pool1_layer_call_and_return_conditional_losses_114019р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_pool1_layer_call_fn_114025р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 2
A__inference_conv3_layer_call_and_return_conditional_losses_114037з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
&__inference_conv3_layer_call_fn_114047з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 2
A__inference_conv4_layer_call_and_return_conditional_losses_114059з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
&__inference_conv4_layer_call_fn_114069з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Љ2І
A__inference_pool2_layer_call_and_return_conditional_losses_114075р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_pool2_layer_call_fn_114081р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_115501Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_flatten_layer_call_fn_115506Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense1_layer_call_and_return_conditional_losses_115517Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense1_layer_call_fn_115526Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense2_layer_call_and_return_conditional_losses_115537Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense2_layer_call_fn_115546Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_output_layer_call_and_return_conditional_losses_115557Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_output_layer_call_fn_115566Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Р2Н
A__inference_norm1_layer_call_and_return_conditional_losses_115602
A__inference_norm1_layer_call_and_return_conditional_losses_115622Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
&__inference_norm1_layer_call_fn_115635
&__inference_norm1_layer_call_fn_115648Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Р2Н
A__inference_norm2_layer_call_and_return_conditional_losses_115684
A__inference_norm2_layer_call_and_return_conditional_losses_115704Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
&__inference_norm2_layer_call_fn_115730
&__inference_norm2_layer_call_fn_115717Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
3B1
$__inference_signature_wrapper_114847input_1І
!__inference__wrapped_model_113968"#()KLJITUSR67<=BC1Ђ.
'Ђ$
"
input_1џџџџџџџџџл 
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџж
A__inference_conv1_layer_call_and_return_conditional_losses_113981IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ў
&__inference_conv1_layer_call_fn_113991IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџж
A__inference_conv2_layer_call_and_return_conditional_losses_114003IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ў
&__inference_conv2_layer_call_fn_114013IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ж
A__inference_conv3_layer_call_and_return_conditional_losses_114037"#IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ў
&__inference_conv3_layer_call_fn_114047"#IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ж
A__inference_conv4_layer_call_and_return_conditional_losses_114059()IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ў
&__inference_conv4_layer_call_fn_114069()IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Є
B__inference_dense1_layer_call_and_return_conditional_losses_115517^670Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 |
'__inference_dense1_layer_call_fn_115526Q670Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЃ
B__inference_dense2_layer_call_and_return_conditional_losses_115537]<=0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 {
'__inference_dense2_layer_call_fn_115546P<=0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Ј
C__inference_flatten_layer_call_and_return_conditional_losses_115501a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџР
 
(__inference_flatten_layer_call_fn_115506T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "џџџџџџџџџРЕ
A__inference_model_layer_call_and_return_conditional_losses_114976p"#()KLJITUSR67<=BC/Ђ,
%Ђ"

Xџџџџџџџџџл 
p
Њ "%Ђ"

0џџџџџџџџџ
 Е
A__inference_model_layer_call_and_return_conditional_losses_115073p"#()KLJITUSR67<=BC/Ђ,
%Ђ"

Xџџџџџџџџџл 
p 
Њ "%Ђ"

0џџџџџџџџџ
 Л
A__inference_model_layer_call_and_return_conditional_losses_115300v"#()KLJITUSR67<=BC5Ђ2
+Ђ(
"
input_1џџџџџџџџџл 
p
Њ "%Ђ"

0џџџџџџџџџ
 Л
A__inference_model_layer_call_and_return_conditional_losses_115397v"#()KLJITUSR67<=BC5Ђ2
+Ђ(
"
input_1џџџџџџџџџл 
p 
Њ "%Ђ"

0џџџџџџџџџ
 
&__inference_model_layer_call_fn_115122c"#()KLJITUSR67<=BC/Ђ,
%Ђ"

Xџџџџџџџџџл 
p
Њ "џџџџџџџџџ
&__inference_model_layer_call_fn_115171c"#()KLJITUSR67<=BC/Ђ,
%Ђ"

Xџџџџџџџџџл 
p 
Њ "џџџџџџџџџ
&__inference_model_layer_call_fn_115446i"#()KLJITUSR67<=BC5Ђ2
+Ђ(
"
input_1џџџџџџџџџл 
p
Њ "џџџџџџџџџ
&__inference_model_layer_call_fn_115495i"#()KLJITUSR67<=BC5Ђ2
+Ђ(
"
input_1џџџџџџџџџл 
p 
Њ "џџџџџџџџџЉ
A__inference_norm1_layer_call_and_return_conditional_losses_115602dKLJI4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p
Њ "&Ђ#

0џџџџџџџџџР
 Љ
A__inference_norm1_layer_call_and_return_conditional_losses_115622dKLJI4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p 
Њ "&Ђ#

0џџџџџџџџџР
 
&__inference_norm1_layer_call_fn_115635WKLJI4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p
Њ "џџџџџџџџџР
&__inference_norm1_layer_call_fn_115648WKLJI4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p 
Њ "џџџџџџџџџРЇ
A__inference_norm2_layer_call_and_return_conditional_losses_115684bTUSR3Ђ0
)Ђ&
 
inputsџџџџџџџџџ[
p
Њ "%Ђ"

0џџџџџџџџџ[
 Ї
A__inference_norm2_layer_call_and_return_conditional_losses_115704bTUSR3Ђ0
)Ђ&
 
inputsџџџџџџџџџ[
p 
Њ "%Ђ"

0џџџџџџџџџ[
 
&__inference_norm2_layer_call_fn_115717UTUSR3Ђ0
)Ђ&
 
inputsџџџџџџџџџ[
p
Њ "џџџџџџџџџ[
&__inference_norm2_layer_call_fn_115730UTUSR3Ђ0
)Ђ&
 
inputsџџџџџџџџџ[
p 
Њ "џџџџџџџџџ[Ђ
B__inference_output_layer_call_and_return_conditional_losses_115557\BC/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 z
'__inference_output_layer_call_fn_115566OBC/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџф
A__inference_pool1_layer_call_and_return_conditional_losses_114019RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
&__inference_pool1_layer_call_fn_114025RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџф
A__inference_pool2_layer_call_and_return_conditional_losses_114075RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
&__inference_pool2_layer_call_fn_114081RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџД
$__inference_signature_wrapper_114847"#()KLJITUSR67<=BC<Ђ9
Ђ 
2Њ/
-
input_1"
input_1џџџџџџџџџл "3Њ0
.
output_1"
output_1џџџџџџџџџ