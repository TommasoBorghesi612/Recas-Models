кн
Щэ
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
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02unknown8йќ
И
model/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namemodel/conv1/kernel
Б
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
И
model/conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namemodel/conv2/kernel
Б
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
И
model/conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_namemodel/conv3/kernel
Б
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
И
model/conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_namemodel/conv4/kernel
Б
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
Д
model/dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЫА*$
shared_namemodel/dense1/kernel
}
'model/dense1/kernel/Read/ReadVariableOpReadVariableOpmodel/dense1/kernel* 
_output_shapes
:
ЫА*
dtype0
{
model/dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_namemodel/dense1/bias
t
%model/dense1/bias/Read/ReadVariableOpReadVariableOpmodel/dense1/bias*
_output_shapes	
:А*
dtype0
Г
model/dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*$
shared_namemodel/dense2/kernel
|
'model/dense2/kernel/Read/ReadVariableOpReadVariableOpmodel/dense2/kernel*
_output_shapes
:	А@*
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
В
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
shape:ј*"
shared_namemodel/norm1/gamma
t
%model/norm1/gamma/Read/ReadVariableOpReadVariableOpmodel/norm1/gamma*
_output_shapes	
:ј*
dtype0
y
model/norm1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*!
shared_namemodel/norm1/beta
r
$model/norm1/beta/Read/ReadVariableOpReadVariableOpmodel/norm1/beta*
_output_shapes	
:ј*
dtype0
З
model/norm1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*(
shared_namemodel/norm1/moving_mean
А
+model/norm1/moving_mean/Read/ReadVariableOpReadVariableOpmodel/norm1/moving_mean*
_output_shapes	
:ј*
dtype0
П
model/norm1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*,
shared_namemodel/norm1/moving_variance
И
/model/norm1/moving_variance/Read/ReadVariableOpReadVariableOpmodel/norm1/moving_variance*
_output_shapes	
:ј*
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
Ж
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
О
model/norm2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:[*,
shared_namemodel/norm2/moving_variance
З
/model/norm2/moving_variance/Read/ReadVariableOpReadVariableOpmodel/norm2/moving_variance*
_output_shapes
:[*
dtype0

NoOpNoOp
Ю4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ў3
valueѕ3Bћ3 B≈3
е
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
h

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
R
.trainable_variables
/regularization_losses
0	variables
1	keras_api
R
2trainable_variables
3regularization_losses
4	variables
5	keras_api
h

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
h

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
h

Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
Ч
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
Ч
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
Ж
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
 
¶
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
≠
Znon_trainable_variables
[layer_metrics
trainable_variables
regularization_losses

\layers
]metrics
^layer_regularization_losses
	variables
 
OM
VARIABLE_VALUEmodel/conv1/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmodel/conv1/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠
_non_trainable_variables
`layer_regularization_losses
trainable_variables
regularization_losses

alayers
bmetrics
clayer_metrics
	variables
OM
VARIABLE_VALUEmodel/conv2/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmodel/conv2/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠
dnon_trainable_variables
elayer_regularization_losses
trainable_variables
regularization_losses

flayers
gmetrics
hlayer_metrics
	variables
 
 
 
≠
inon_trainable_variables
jlayer_regularization_losses
trainable_variables
regularization_losses

klayers
lmetrics
mlayer_metrics
 	variables
OM
VARIABLE_VALUEmodel/conv3/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmodel/conv3/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
≠
nnon_trainable_variables
olayer_regularization_losses
$trainable_variables
%regularization_losses

players
qmetrics
rlayer_metrics
&	variables
OM
VARIABLE_VALUEmodel/conv4/kernel'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmodel/conv4/bias%conv4/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
≠
snon_trainable_variables
tlayer_regularization_losses
*trainable_variables
+regularization_losses

ulayers
vmetrics
wlayer_metrics
,	variables
 
 
 
≠
xnon_trainable_variables
ylayer_regularization_losses
.trainable_variables
/regularization_losses

zlayers
{metrics
|layer_metrics
0	variables
 
 
 
ѓ
}non_trainable_variables
~layer_regularization_losses
2trainable_variables
3regularization_losses

layers
Аmetrics
Бlayer_metrics
4	variables
MK
VARIABLE_VALUEmodel/dense1/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEmodel/dense1/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
≤
Вnon_trainable_variables
 Гlayer_regularization_losses
8trainable_variables
9regularization_losses
Дlayers
Еmetrics
Жlayer_metrics
:	variables
MK
VARIABLE_VALUEmodel/dense2/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEmodel/dense2/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
 

<0
=1
≤
Зnon_trainable_variables
 Иlayer_regularization_losses
>trainable_variables
?regularization_losses
Йlayers
Кmetrics
Лlayer_metrics
@	variables
MK
VARIABLE_VALUEmodel/output/kernel$d3/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEmodel/output/bias"d3/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
≤
Мnon_trainable_variables
 Нlayer_regularization_losses
Dtrainable_variables
Eregularization_losses
Оlayers
Пmetrics
Рlayer_metrics
F	variables
 
PN
VARIABLE_VALUEmodel/norm1/gamma)b_norm_1/gamma/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEmodel/norm1/beta(b_norm_1/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEmodel/norm1/moving_mean/b_norm_1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodel/norm1/moving_variance3b_norm_1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 

I0
J1
K2
L3
≤
Сnon_trainable_variables
 Тlayer_regularization_losses
Mtrainable_variables
Nregularization_losses
Уlayers
Фmetrics
Хlayer_metrics
O	variables
 
PN
VARIABLE_VALUEmodel/norm2/gamma)b_norm_2/gamma/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEmodel/norm2/beta(b_norm_2/beta/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEmodel/norm2/moving_mean/b_norm_2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodel/norm2/moving_variance3b_norm_2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

R0
S1
 

R0
S1
T2
U3
≤
Цnon_trainable_variables
 Чlayer_regularization_losses
Vtrainable_variables
Wregularization_losses
Шlayers
Щmetrics
Ъlayer_metrics
X	variables

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
:€€€€€€€€€џ *
dtype0*
shape:€€€€€€€€€џ 
Д
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1model/conv1/kernelmodel/conv1/biasmodel/conv2/kernelmodel/conv2/biasmodel/conv3/kernelmodel/conv3/biasmodel/conv4/kernelmodel/conv4/biasmodel/norm1/moving_meanmodel/norm1/moving_variancemodel/norm1/betamodel/norm1/gammamodel/norm2/moving_meanmodel/norm2/moving_variancemodel/norm2/betamodel/norm2/gammamodel/dense1/kernelmodel/dense1/biasmodel/dense2/kernelmodel/dense2/biasmodel/output/kernelmodel/output/bias*"
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference_signature_wrapper_87187
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
П	
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
CPU

GPU2*0J 8*'
f"R 
__inference__traced_save_88163
“
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
CPU

GPU2*0J 8**
f%R#
!__inference__traced_restore_88241ТЎ

р
Ш
%__inference_norm1_layer_call_fn_87975

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_norm1_layer_call_and_return_conditional_losses_865172
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ј::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ј
 
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
л
©
A__inference_dense1_layer_call_and_return_conditional_losses_86830

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЫА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Ы:::P L
(
_output_shapes
:€€€€€€€€€Ы
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ї
^
B__inference_flatten_layer_call_and_return_conditional_losses_87841

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
оЪ
Щ	
@__inference_model_layer_call_and_return_conditional_losses_87316
input_1(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource
norm1_assignmovingavg_87236!
norm1_assignmovingavg_1_87242&
"norm1_cast_readvariableop_resource(
$norm1_cast_1_readvariableop_resource
norm2_assignmovingavg_87268!
norm2_assignmovingavg_1_87274&
"norm2_cast_readvariableop_resource(
$norm2_cast_1_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityИҐ)norm1/AssignMovingAvg/AssignSubVariableOpҐ+norm1/AssignMovingAvg_1/AssignSubVariableOpҐ)norm2/AssignMovingAvg/AssignSubVariableOpҐ+norm2/AssignMovingAvg_1/AssignSubVariableOp_
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
split/split_dimЃ
splitSplitVinput_1Const:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':€€€€€€€€€А :€€€€€€€€€[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2	
ReshapeІ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpј
conv1/Conv2DConv2DReshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp†
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

conv1/ReluІ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOp»
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp†
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

conv2/Reluѓ
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingSAME*
strides
2
pool1/MaxPoolІ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOp≈
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
conv3/Conv2DЮ
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp†
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

conv3/ReluІ
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv4/Conv2D/ReadVariableOp«
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv4/Conv2DЮ
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOp†
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

conv4/Reluѓ
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
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
valueB"€€€€@  2
flatten/ConstР
flatten/ReshapeReshapepool2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
flatten/ReshapeЦ
$norm1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2&
$norm1/moments/mean/reduction_indicesі
norm1/moments/meanMeanflatten/Reshape:output:0-norm1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ј*
	keep_dims(2
norm1/moments/meanП
norm1/moments/StopGradientStopGradientnorm1/moments/mean:output:0*
T0*
_output_shapes
:	ј2
norm1/moments/StopGradient…
norm1/moments/SquaredDifferenceSquaredDifferenceflatten/Reshape:output:0#norm1/moments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2!
norm1/moments/SquaredDifferenceЮ
(norm1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2*
(norm1/moments/variance/reduction_indicesЋ
norm1/moments/varianceMean#norm1/moments/SquaredDifference:z:01norm1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ј*
	keep_dims(2
norm1/moments/varianceУ
norm1/moments/SqueezeSqueezenorm1/moments/mean:output:0*
T0*
_output_shapes	
:ј*
squeeze_dims
 2
norm1/moments/SqueezeЫ
norm1/moments/Squeeze_1Squeezenorm1/moments/variance:output:0*
T0*
_output_shapes	
:ј*
squeeze_dims
 2
norm1/moments/Squeeze_1ѓ
norm1/AssignMovingAvg/decayConst*.
_class$
" loc:@norm1/AssignMovingAvg/87236*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
norm1/AssignMovingAvg/decay•
$norm1/AssignMovingAvg/ReadVariableOpReadVariableOpnorm1_assignmovingavg_87236*
_output_shapes	
:ј*
dtype02&
$norm1/AssignMovingAvg/ReadVariableOpб
norm1/AssignMovingAvg/subSub,norm1/AssignMovingAvg/ReadVariableOp:value:0norm1/moments/Squeeze:output:0*
T0*.
_class$
" loc:@norm1/AssignMovingAvg/87236*
_output_shapes	
:ј2
norm1/AssignMovingAvg/subЎ
norm1/AssignMovingAvg/mulMulnorm1/AssignMovingAvg/sub:z:0$norm1/AssignMovingAvg/decay:output:0*
T0*.
_class$
" loc:@norm1/AssignMovingAvg/87236*
_output_shapes	
:ј2
norm1/AssignMovingAvg/mul£
)norm1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpnorm1_assignmovingavg_87236norm1/AssignMovingAvg/mul:z:0%^norm1/AssignMovingAvg/ReadVariableOp*.
_class$
" loc:@norm1/AssignMovingAvg/87236*
_output_shapes
 *
dtype02+
)norm1/AssignMovingAvg/AssignSubVariableOpµ
norm1/AssignMovingAvg_1/decayConst*0
_class&
$"loc:@norm1/AssignMovingAvg_1/87242*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
norm1/AssignMovingAvg_1/decayЂ
&norm1/AssignMovingAvg_1/ReadVariableOpReadVariableOpnorm1_assignmovingavg_1_87242*
_output_shapes	
:ј*
dtype02(
&norm1/AssignMovingAvg_1/ReadVariableOpл
norm1/AssignMovingAvg_1/subSub.norm1/AssignMovingAvg_1/ReadVariableOp:value:0 norm1/moments/Squeeze_1:output:0*
T0*0
_class&
$"loc:@norm1/AssignMovingAvg_1/87242*
_output_shapes	
:ј2
norm1/AssignMovingAvg_1/subв
norm1/AssignMovingAvg_1/mulMulnorm1/AssignMovingAvg_1/sub:z:0&norm1/AssignMovingAvg_1/decay:output:0*
T0*0
_class&
$"loc:@norm1/AssignMovingAvg_1/87242*
_output_shapes	
:ј2
norm1/AssignMovingAvg_1/mulѓ
+norm1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpnorm1_assignmovingavg_1_87242norm1/AssignMovingAvg_1/mul:z:0'^norm1/AssignMovingAvg_1/ReadVariableOp*0
_class&
$"loc:@norm1/AssignMovingAvg_1/87242*
_output_shapes
 *
dtype02-
+norm1/AssignMovingAvg_1/AssignSubVariableOpЦ
norm1/Cast/ReadVariableOpReadVariableOp"norm1_cast_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast/ReadVariableOpЬ
norm1/Cast_1/ReadVariableOpReadVariableOp$norm1_cast_1_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast_1/ReadVariableOps
norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
norm1/batchnorm/add/yЫ
norm1/batchnorm/addAddV2 norm1/moments/Squeeze_1:output:0norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/addv
norm1/batchnorm/RsqrtRsqrtnorm1/batchnorm/add:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/RsqrtЧ
norm1/batchnorm/mulMulnorm1/batchnorm/Rsqrt:y:0#norm1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/mulЫ
norm1/batchnorm/mul_1Mulflatten/Reshape:output:0norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
norm1/batchnorm/mul_1Ф
norm1/batchnorm/mul_2Mulnorm1/moments/Squeeze:output:0norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/mul_2Х
norm1/batchnorm/subSub!norm1/Cast/ReadVariableOp:value:0norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/subЮ
norm1/batchnorm/add_1AddV2norm1/batchnorm/mul_1:z:0norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
norm1/batchnorm/add_1Ц
$norm2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2&
$norm2/moments/mean/reduction_indices©
norm2/moments/meanMeansplit:output:1-norm2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
norm2/moments/meanО
norm2/moments/StopGradientStopGradientnorm2/moments/mean:output:0*
T0*
_output_shapes

:[2
norm2/moments/StopGradientЊ
norm2/moments/SquaredDifferenceSquaredDifferencesplit:output:1#norm2/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€[2!
norm2/moments/SquaredDifferenceЮ
(norm2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2*
(norm2/moments/variance/reduction_indices 
norm2/moments/varianceMean#norm2/moments/SquaredDifference:z:01norm2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
norm2/moments/varianceТ
norm2/moments/SqueezeSqueezenorm2/moments/mean:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
norm2/moments/SqueezeЪ
norm2/moments/Squeeze_1Squeezenorm2/moments/variance:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
norm2/moments/Squeeze_1ѓ
norm2/AssignMovingAvg/decayConst*.
_class$
" loc:@norm2/AssignMovingAvg/87268*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
norm2/AssignMovingAvg/decay§
$norm2/AssignMovingAvg/ReadVariableOpReadVariableOpnorm2_assignmovingavg_87268*
_output_shapes
:[*
dtype02&
$norm2/AssignMovingAvg/ReadVariableOpа
norm2/AssignMovingAvg/subSub,norm2/AssignMovingAvg/ReadVariableOp:value:0norm2/moments/Squeeze:output:0*
T0*.
_class$
" loc:@norm2/AssignMovingAvg/87268*
_output_shapes
:[2
norm2/AssignMovingAvg/sub„
norm2/AssignMovingAvg/mulMulnorm2/AssignMovingAvg/sub:z:0$norm2/AssignMovingAvg/decay:output:0*
T0*.
_class$
" loc:@norm2/AssignMovingAvg/87268*
_output_shapes
:[2
norm2/AssignMovingAvg/mul£
)norm2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpnorm2_assignmovingavg_87268norm2/AssignMovingAvg/mul:z:0%^norm2/AssignMovingAvg/ReadVariableOp*.
_class$
" loc:@norm2/AssignMovingAvg/87268*
_output_shapes
 *
dtype02+
)norm2/AssignMovingAvg/AssignSubVariableOpµ
norm2/AssignMovingAvg_1/decayConst*0
_class&
$"loc:@norm2/AssignMovingAvg_1/87274*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
norm2/AssignMovingAvg_1/decay™
&norm2/AssignMovingAvg_1/ReadVariableOpReadVariableOpnorm2_assignmovingavg_1_87274*
_output_shapes
:[*
dtype02(
&norm2/AssignMovingAvg_1/ReadVariableOpк
norm2/AssignMovingAvg_1/subSub.norm2/AssignMovingAvg_1/ReadVariableOp:value:0 norm2/moments/Squeeze_1:output:0*
T0*0
_class&
$"loc:@norm2/AssignMovingAvg_1/87274*
_output_shapes
:[2
norm2/AssignMovingAvg_1/subб
norm2/AssignMovingAvg_1/mulMulnorm2/AssignMovingAvg_1/sub:z:0&norm2/AssignMovingAvg_1/decay:output:0*
T0*0
_class&
$"loc:@norm2/AssignMovingAvg_1/87274*
_output_shapes
:[2
norm2/AssignMovingAvg_1/mulѓ
+norm2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpnorm2_assignmovingavg_1_87274norm2/AssignMovingAvg_1/mul:z:0'^norm2/AssignMovingAvg_1/ReadVariableOp*0
_class&
$"loc:@norm2/AssignMovingAvg_1/87274*
_output_shapes
 *
dtype02-
+norm2/AssignMovingAvg_1/AssignSubVariableOpХ
norm2/Cast/ReadVariableOpReadVariableOp"norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast/ReadVariableOpЫ
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
 *oГ:2
norm2/batchnorm/add/yЪ
norm2/batchnorm/addAddV2 norm2/moments/Squeeze_1:output:0norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
norm2/batchnorm/addu
norm2/batchnorm/RsqrtRsqrtnorm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/RsqrtЦ
norm2/batchnorm/mulMulnorm2/batchnorm/Rsqrt:y:0#norm2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mulР
norm2/batchnorm/mul_1Mulsplit:output:1norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
norm2/batchnorm/mul_1У
norm2/batchnorm/mul_2Mulnorm2/moments/Squeeze:output:0norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul_2Ф
norm2/batchnorm/subSub!norm2/Cast/ReadVariableOp:value:0norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/subЭ
norm2/batchnorm/add_1AddV2norm2/batchnorm/mul_1:z:0norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
norm2/batchnorm/add_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis§
concatConcatV2norm1/batchnorm/add_1:z:0norm2/batchnorm/add_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ы2
concat§
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
ЫА*
dtype02
dense1/MatMul/ReadVariableOpТ
dense1/MatMulMatMulconcat:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/MatMulҐ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense1/BiasAdd/ReadVariableOpЮ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/Relu£
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense2/MatMul/ReadVariableOpЫ
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/MatMul°
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense2/BiasAdd/ReadVariableOpЭ
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/ReluҐ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOpЫ
output/MatMulMatMuldense2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/MatMul°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/Relu°
IdentityIdentityoutput/Relu:activations:0*^norm1/AssignMovingAvg/AssignSubVariableOp,^norm1/AssignMovingAvg_1/AssignSubVariableOp*^norm2/AssignMovingAvg/AssignSubVariableOp,^norm2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ ::::::::::::::::::::::2V
)norm1/AssignMovingAvg/AssignSubVariableOp)norm1/AssignMovingAvg/AssignSubVariableOp2Z
+norm1/AssignMovingAvg_1/AssignSubVariableOp+norm1/AssignMovingAvg_1/AssignSubVariableOp2V
)norm2/AssignMovingAvg/AssignSubVariableOp)norm2/AssignMovingAvg/AssignSubVariableOp2Z
+norm2/AssignMovingAvg_1/AssignSubVariableOp+norm2/AssignMovingAvg_1/AssignSubVariableOp:Q M
(
_output_shapes
:€€€€€€€€€џ 
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
х
\
@__inference_pool1_layer_call_and_return_conditional_losses_86359

inputs
identityђ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
≤
%__inference_model_layer_call_fn_87462
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
identityИҐStatefulPartitionedCallб
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
:€€€€€€€€€*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_870402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€џ 
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
сb
Е
@__inference_model_layer_call_and_return_conditional_losses_87413
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
identityИ_
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
split/split_dimЃ
splitSplitVinput_1Const:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':€€€€€€€€€А :€€€€€€€€€[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2	
ReshapeІ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpј
conv1/Conv2DConv2DReshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp†
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

conv1/ReluІ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOp»
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp†
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

conv2/Reluѓ
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingSAME*
strides
2
pool1/MaxPoolІ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOp≈
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
conv3/Conv2DЮ
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp†
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

conv3/ReluІ
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv4/Conv2D/ReadVariableOp«
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv4/Conv2DЮ
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOp†
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

conv4/Reluѓ
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
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
valueB"€€€€@  2
flatten/ConstР
flatten/ReshapeReshapepool2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
flatten/ReshapeЦ
norm1/Cast/ReadVariableOpReadVariableOp"norm1_cast_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast/ReadVariableOpЬ
norm1/Cast_1/ReadVariableOpReadVariableOp$norm1_cast_1_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast_1/ReadVariableOpЬ
norm1/Cast_2/ReadVariableOpReadVariableOp$norm1_cast_2_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast_2/ReadVariableOpЬ
norm1/Cast_3/ReadVariableOpReadVariableOp$norm1_cast_3_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast_3/ReadVariableOps
norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
norm1/batchnorm/add/yЮ
norm1/batchnorm/addAddV2#norm1/Cast_1/ReadVariableOp:value:0norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/addv
norm1/batchnorm/RsqrtRsqrtnorm1/batchnorm/add:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/RsqrtЧ
norm1/batchnorm/mulMulnorm1/batchnorm/Rsqrt:y:0#norm1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/mulЫ
norm1/batchnorm/mul_1Mulflatten/Reshape:output:0norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
norm1/batchnorm/mul_1Ч
norm1/batchnorm/mul_2Mul!norm1/Cast/ReadVariableOp:value:0norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/mul_2Ч
norm1/batchnorm/subSub#norm1/Cast_2/ReadVariableOp:value:0norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/subЮ
norm1/batchnorm/add_1AddV2norm1/batchnorm/mul_1:z:0norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
norm1/batchnorm/add_1Х
norm2/Cast/ReadVariableOpReadVariableOp"norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast/ReadVariableOpЫ
norm2/Cast_1/ReadVariableOpReadVariableOp$norm2_cast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_1/ReadVariableOpЫ
norm2/Cast_2/ReadVariableOpReadVariableOp$norm2_cast_2_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_2/ReadVariableOpЫ
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
 *oГ:2
norm2/batchnorm/add/yЭ
norm2/batchnorm/addAddV2#norm2/Cast_1/ReadVariableOp:value:0norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
norm2/batchnorm/addu
norm2/batchnorm/RsqrtRsqrtnorm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/RsqrtЦ
norm2/batchnorm/mulMulnorm2/batchnorm/Rsqrt:y:0#norm2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mulР
norm2/batchnorm/mul_1Mulsplit:output:1norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
norm2/batchnorm/mul_1Ц
norm2/batchnorm/mul_2Mul!norm2/Cast/ReadVariableOp:value:0norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul_2Ц
norm2/batchnorm/subSub#norm2/Cast_2/ReadVariableOp:value:0norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/subЭ
norm2/batchnorm/add_1AddV2norm2/batchnorm/mul_1:z:0norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
norm2/batchnorm/add_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis§
concatConcatV2norm1/batchnorm/add_1:z:0norm2/batchnorm/add_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ы2
concat§
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
ЫА*
dtype02
dense1/MatMul/ReadVariableOpТ
dense1/MatMulMatMulconcat:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/MatMulҐ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense1/BiasAdd/ReadVariableOpЮ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/Relu£
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense2/MatMul/ReadVariableOpЫ
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/MatMul°
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense2/BiasAdd/ReadVariableOpЭ
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/ReluҐ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOpЫ
output/MatMulMatMuldense2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/MatMul°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/Relum
IdentityIdentityoutput/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ :::::::::::::::::::::::Q M
(
_output_shapes
:€€€€€€€€€џ 
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
ѓ

®
@__inference_conv1_layer_call_and_return_conditional_losses_86321

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
в
©
A__inference_output_layer_call_and_return_conditional_losses_86884

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Б
C
'__inference_flatten_layer_call_fn_87846

inputs
identityҐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_867392
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
№Ъ
У	
@__inference_model_layer_call_and_return_conditional_losses_87640
x(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource
norm1_assignmovingavg_87560!
norm1_assignmovingavg_1_87566&
"norm1_cast_readvariableop_resource(
$norm1_cast_1_readvariableop_resource
norm2_assignmovingavg_87592!
norm2_assignmovingavg_1_87598&
"norm2_cast_readvariableop_resource(
$norm2_cast_1_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityИҐ)norm1/AssignMovingAvg/AssignSubVariableOpҐ+norm1/AssignMovingAvg_1/AssignSubVariableOpҐ)norm2/AssignMovingAvg/AssignSubVariableOpҐ+norm2/AssignMovingAvg_1/AssignSubVariableOp_
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
split/split_dim®
splitSplitVxConst:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':€€€€€€€€€А :€€€€€€€€€[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2	
ReshapeІ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpј
conv1/Conv2DConv2DReshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp†
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

conv1/ReluІ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOp»
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp†
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

conv2/Reluѓ
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingSAME*
strides
2
pool1/MaxPoolІ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOp≈
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
conv3/Conv2DЮ
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp†
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

conv3/ReluІ
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv4/Conv2D/ReadVariableOp«
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv4/Conv2DЮ
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOp†
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

conv4/Reluѓ
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
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
valueB"€€€€@  2
flatten/ConstР
flatten/ReshapeReshapepool2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
flatten/ReshapeЦ
$norm1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2&
$norm1/moments/mean/reduction_indicesі
norm1/moments/meanMeanflatten/Reshape:output:0-norm1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ј*
	keep_dims(2
norm1/moments/meanП
norm1/moments/StopGradientStopGradientnorm1/moments/mean:output:0*
T0*
_output_shapes
:	ј2
norm1/moments/StopGradient…
norm1/moments/SquaredDifferenceSquaredDifferenceflatten/Reshape:output:0#norm1/moments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2!
norm1/moments/SquaredDifferenceЮ
(norm1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2*
(norm1/moments/variance/reduction_indicesЋ
norm1/moments/varianceMean#norm1/moments/SquaredDifference:z:01norm1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ј*
	keep_dims(2
norm1/moments/varianceУ
norm1/moments/SqueezeSqueezenorm1/moments/mean:output:0*
T0*
_output_shapes	
:ј*
squeeze_dims
 2
norm1/moments/SqueezeЫ
norm1/moments/Squeeze_1Squeezenorm1/moments/variance:output:0*
T0*
_output_shapes	
:ј*
squeeze_dims
 2
norm1/moments/Squeeze_1ѓ
norm1/AssignMovingAvg/decayConst*.
_class$
" loc:@norm1/AssignMovingAvg/87560*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
norm1/AssignMovingAvg/decay•
$norm1/AssignMovingAvg/ReadVariableOpReadVariableOpnorm1_assignmovingavg_87560*
_output_shapes	
:ј*
dtype02&
$norm1/AssignMovingAvg/ReadVariableOpб
norm1/AssignMovingAvg/subSub,norm1/AssignMovingAvg/ReadVariableOp:value:0norm1/moments/Squeeze:output:0*
T0*.
_class$
" loc:@norm1/AssignMovingAvg/87560*
_output_shapes	
:ј2
norm1/AssignMovingAvg/subЎ
norm1/AssignMovingAvg/mulMulnorm1/AssignMovingAvg/sub:z:0$norm1/AssignMovingAvg/decay:output:0*
T0*.
_class$
" loc:@norm1/AssignMovingAvg/87560*
_output_shapes	
:ј2
norm1/AssignMovingAvg/mul£
)norm1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpnorm1_assignmovingavg_87560norm1/AssignMovingAvg/mul:z:0%^norm1/AssignMovingAvg/ReadVariableOp*.
_class$
" loc:@norm1/AssignMovingAvg/87560*
_output_shapes
 *
dtype02+
)norm1/AssignMovingAvg/AssignSubVariableOpµ
norm1/AssignMovingAvg_1/decayConst*0
_class&
$"loc:@norm1/AssignMovingAvg_1/87566*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
norm1/AssignMovingAvg_1/decayЂ
&norm1/AssignMovingAvg_1/ReadVariableOpReadVariableOpnorm1_assignmovingavg_1_87566*
_output_shapes	
:ј*
dtype02(
&norm1/AssignMovingAvg_1/ReadVariableOpл
norm1/AssignMovingAvg_1/subSub.norm1/AssignMovingAvg_1/ReadVariableOp:value:0 norm1/moments/Squeeze_1:output:0*
T0*0
_class&
$"loc:@norm1/AssignMovingAvg_1/87566*
_output_shapes	
:ј2
norm1/AssignMovingAvg_1/subв
norm1/AssignMovingAvg_1/mulMulnorm1/AssignMovingAvg_1/sub:z:0&norm1/AssignMovingAvg_1/decay:output:0*
T0*0
_class&
$"loc:@norm1/AssignMovingAvg_1/87566*
_output_shapes	
:ј2
norm1/AssignMovingAvg_1/mulѓ
+norm1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpnorm1_assignmovingavg_1_87566norm1/AssignMovingAvg_1/mul:z:0'^norm1/AssignMovingAvg_1/ReadVariableOp*0
_class&
$"loc:@norm1/AssignMovingAvg_1/87566*
_output_shapes
 *
dtype02-
+norm1/AssignMovingAvg_1/AssignSubVariableOpЦ
norm1/Cast/ReadVariableOpReadVariableOp"norm1_cast_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast/ReadVariableOpЬ
norm1/Cast_1/ReadVariableOpReadVariableOp$norm1_cast_1_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast_1/ReadVariableOps
norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
norm1/batchnorm/add/yЫ
norm1/batchnorm/addAddV2 norm1/moments/Squeeze_1:output:0norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/addv
norm1/batchnorm/RsqrtRsqrtnorm1/batchnorm/add:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/RsqrtЧ
norm1/batchnorm/mulMulnorm1/batchnorm/Rsqrt:y:0#norm1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/mulЫ
norm1/batchnorm/mul_1Mulflatten/Reshape:output:0norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
norm1/batchnorm/mul_1Ф
norm1/batchnorm/mul_2Mulnorm1/moments/Squeeze:output:0norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/mul_2Х
norm1/batchnorm/subSub!norm1/Cast/ReadVariableOp:value:0norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/subЮ
norm1/batchnorm/add_1AddV2norm1/batchnorm/mul_1:z:0norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
norm1/batchnorm/add_1Ц
$norm2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2&
$norm2/moments/mean/reduction_indices©
norm2/moments/meanMeansplit:output:1-norm2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
norm2/moments/meanО
norm2/moments/StopGradientStopGradientnorm2/moments/mean:output:0*
T0*
_output_shapes

:[2
norm2/moments/StopGradientЊ
norm2/moments/SquaredDifferenceSquaredDifferencesplit:output:1#norm2/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€[2!
norm2/moments/SquaredDifferenceЮ
(norm2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2*
(norm2/moments/variance/reduction_indices 
norm2/moments/varianceMean#norm2/moments/SquaredDifference:z:01norm2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
norm2/moments/varianceТ
norm2/moments/SqueezeSqueezenorm2/moments/mean:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
norm2/moments/SqueezeЪ
norm2/moments/Squeeze_1Squeezenorm2/moments/variance:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
norm2/moments/Squeeze_1ѓ
norm2/AssignMovingAvg/decayConst*.
_class$
" loc:@norm2/AssignMovingAvg/87592*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
norm2/AssignMovingAvg/decay§
$norm2/AssignMovingAvg/ReadVariableOpReadVariableOpnorm2_assignmovingavg_87592*
_output_shapes
:[*
dtype02&
$norm2/AssignMovingAvg/ReadVariableOpа
norm2/AssignMovingAvg/subSub,norm2/AssignMovingAvg/ReadVariableOp:value:0norm2/moments/Squeeze:output:0*
T0*.
_class$
" loc:@norm2/AssignMovingAvg/87592*
_output_shapes
:[2
norm2/AssignMovingAvg/sub„
norm2/AssignMovingAvg/mulMulnorm2/AssignMovingAvg/sub:z:0$norm2/AssignMovingAvg/decay:output:0*
T0*.
_class$
" loc:@norm2/AssignMovingAvg/87592*
_output_shapes
:[2
norm2/AssignMovingAvg/mul£
)norm2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpnorm2_assignmovingavg_87592norm2/AssignMovingAvg/mul:z:0%^norm2/AssignMovingAvg/ReadVariableOp*.
_class$
" loc:@norm2/AssignMovingAvg/87592*
_output_shapes
 *
dtype02+
)norm2/AssignMovingAvg/AssignSubVariableOpµ
norm2/AssignMovingAvg_1/decayConst*0
_class&
$"loc:@norm2/AssignMovingAvg_1/87598*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
norm2/AssignMovingAvg_1/decay™
&norm2/AssignMovingAvg_1/ReadVariableOpReadVariableOpnorm2_assignmovingavg_1_87598*
_output_shapes
:[*
dtype02(
&norm2/AssignMovingAvg_1/ReadVariableOpк
norm2/AssignMovingAvg_1/subSub.norm2/AssignMovingAvg_1/ReadVariableOp:value:0 norm2/moments/Squeeze_1:output:0*
T0*0
_class&
$"loc:@norm2/AssignMovingAvg_1/87598*
_output_shapes
:[2
norm2/AssignMovingAvg_1/subб
norm2/AssignMovingAvg_1/mulMulnorm2/AssignMovingAvg_1/sub:z:0&norm2/AssignMovingAvg_1/decay:output:0*
T0*0
_class&
$"loc:@norm2/AssignMovingAvg_1/87598*
_output_shapes
:[2
norm2/AssignMovingAvg_1/mulѓ
+norm2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpnorm2_assignmovingavg_1_87598norm2/AssignMovingAvg_1/mul:z:0'^norm2/AssignMovingAvg_1/ReadVariableOp*0
_class&
$"loc:@norm2/AssignMovingAvg_1/87598*
_output_shapes
 *
dtype02-
+norm2/AssignMovingAvg_1/AssignSubVariableOpХ
norm2/Cast/ReadVariableOpReadVariableOp"norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast/ReadVariableOpЫ
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
 *oГ:2
norm2/batchnorm/add/yЪ
norm2/batchnorm/addAddV2 norm2/moments/Squeeze_1:output:0norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
norm2/batchnorm/addu
norm2/batchnorm/RsqrtRsqrtnorm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/RsqrtЦ
norm2/batchnorm/mulMulnorm2/batchnorm/Rsqrt:y:0#norm2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mulР
norm2/batchnorm/mul_1Mulsplit:output:1norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
norm2/batchnorm/mul_1У
norm2/batchnorm/mul_2Mulnorm2/moments/Squeeze:output:0norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul_2Ф
norm2/batchnorm/subSub!norm2/Cast/ReadVariableOp:value:0norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/subЭ
norm2/batchnorm/add_1AddV2norm2/batchnorm/mul_1:z:0norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
norm2/batchnorm/add_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis§
concatConcatV2norm1/batchnorm/add_1:z:0norm2/batchnorm/add_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ы2
concat§
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
ЫА*
dtype02
dense1/MatMul/ReadVariableOpТ
dense1/MatMulMatMulconcat:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/MatMulҐ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense1/BiasAdd/ReadVariableOpЮ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/Relu£
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense2/MatMul/ReadVariableOpЫ
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/MatMul°
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense2/BiasAdd/ReadVariableOpЭ
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/ReluҐ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOpЫ
output/MatMulMatMuldense2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/MatMul°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/Relu°
IdentityIdentityoutput/Relu:activations:0*^norm1/AssignMovingAvg/AssignSubVariableOp,^norm1/AssignMovingAvg_1/AssignSubVariableOp*^norm2/AssignMovingAvg/AssignSubVariableOp,^norm2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ ::::::::::::::::::::::2V
)norm1/AssignMovingAvg/AssignSubVariableOp)norm1/AssignMovingAvg/AssignSubVariableOp2Z
+norm1/AssignMovingAvg_1/AssignSubVariableOp+norm1/AssignMovingAvg_1/AssignSubVariableOp2V
)norm2/AssignMovingAvg/AssignSubVariableOp)norm2/AssignMovingAvg/AssignSubVariableOp2Z
+norm2/AssignMovingAvg_1/AssignSubVariableOp+norm2/AssignMovingAvg_1/AssignSubVariableOp:K G
(
_output_shapes
:€€€€€€€€€џ 
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
м
Ш
%__inference_norm2_layer_call_fn_88057

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€[*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_norm2_layer_call_and_return_conditional_losses_866572
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€[::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€[
 
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
ч
{
&__inference_dense1_layer_call_fn_87866

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall”
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_868302
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Ы::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Ы
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ў
z
%__inference_conv4_layer_call_fn_86409

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_863992
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ѓ

®
@__inference_conv4_layer_call_and_return_conditional_losses_86399

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
З
н
@__inference_norm1_layer_call_and_return_conditional_losses_86550

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityИД
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast_1/ReadVariableOpК
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast_2/ReadVariableOpК
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЖ
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ј2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ј2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ј2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ј2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ј2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ј:::::P L
(
_output_shapes
:€€€€€€€€€ј
 
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
З
н
@__inference_norm1_layer_call_and_return_conditional_losses_87962

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityИД
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast_1/ReadVariableOpК
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast_2/ReadVariableOpК
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЖ
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ј2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ј2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ј2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ј2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ј2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ј:::::P L
(
_output_shapes
:€€€€€€€€€ј
 
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
”
ђ
%__inference_model_layer_call_fn_87835
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
identityИҐStatefulPartitionedCallџ
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
:€€€€€€€€€*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_870402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:€€€€€€€€€џ 
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
ш
A
%__inference_pool1_layer_call_fn_86365

inputs
identity¬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_pool1_layer_call_and_return_conditional_losses_863592
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
яb
€
@__inference_model_layer_call_and_return_conditional_losses_87737
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
identityИ_
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
split/split_dim®
splitSplitVxConst:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':€€€€€€€€€А :€€€€€€€€€[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2	
ReshapeІ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpј
conv1/Conv2DConv2DReshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp†
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

conv1/ReluІ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOp»
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp†
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

conv2/Reluѓ
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingSAME*
strides
2
pool1/MaxPoolІ
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOp≈
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
conv3/Conv2DЮ
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp†
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

conv3/ReluІ
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv4/Conv2D/ReadVariableOp«
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv4/Conv2DЮ
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOp†
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

conv4/Reluѓ
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
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
valueB"€€€€@  2
flatten/ConstР
flatten/ReshapeReshapepool2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
flatten/ReshapeЦ
norm1/Cast/ReadVariableOpReadVariableOp"norm1_cast_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast/ReadVariableOpЬ
norm1/Cast_1/ReadVariableOpReadVariableOp$norm1_cast_1_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast_1/ReadVariableOpЬ
norm1/Cast_2/ReadVariableOpReadVariableOp$norm1_cast_2_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast_2/ReadVariableOpЬ
norm1/Cast_3/ReadVariableOpReadVariableOp$norm1_cast_3_readvariableop_resource*
_output_shapes	
:ј*
dtype02
norm1/Cast_3/ReadVariableOps
norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
norm1/batchnorm/add/yЮ
norm1/batchnorm/addAddV2#norm1/Cast_1/ReadVariableOp:value:0norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/addv
norm1/batchnorm/RsqrtRsqrtnorm1/batchnorm/add:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/RsqrtЧ
norm1/batchnorm/mulMulnorm1/batchnorm/Rsqrt:y:0#norm1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/mulЫ
norm1/batchnorm/mul_1Mulflatten/Reshape:output:0norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
norm1/batchnorm/mul_1Ч
norm1/batchnorm/mul_2Mul!norm1/Cast/ReadVariableOp:value:0norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/mul_2Ч
norm1/batchnorm/subSub#norm1/Cast_2/ReadVariableOp:value:0norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ј2
norm1/batchnorm/subЮ
norm1/batchnorm/add_1AddV2norm1/batchnorm/mul_1:z:0norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
norm1/batchnorm/add_1Х
norm2/Cast/ReadVariableOpReadVariableOp"norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast/ReadVariableOpЫ
norm2/Cast_1/ReadVariableOpReadVariableOp$norm2_cast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_1/ReadVariableOpЫ
norm2/Cast_2/ReadVariableOpReadVariableOp$norm2_cast_2_readvariableop_resource*
_output_shapes
:[*
dtype02
norm2/Cast_2/ReadVariableOpЫ
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
 *oГ:2
norm2/batchnorm/add/yЭ
norm2/batchnorm/addAddV2#norm2/Cast_1/ReadVariableOp:value:0norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
norm2/batchnorm/addu
norm2/batchnorm/RsqrtRsqrtnorm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/RsqrtЦ
norm2/batchnorm/mulMulnorm2/batchnorm/Rsqrt:y:0#norm2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mulР
norm2/batchnorm/mul_1Mulsplit:output:1norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
norm2/batchnorm/mul_1Ц
norm2/batchnorm/mul_2Mul!norm2/Cast/ReadVariableOp:value:0norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/mul_2Ц
norm2/batchnorm/subSub#norm2/Cast_2/ReadVariableOp:value:0norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
norm2/batchnorm/subЭ
norm2/batchnorm/add_1AddV2norm2/batchnorm/mul_1:z:0norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
norm2/batchnorm/add_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis§
concatConcatV2norm1/batchnorm/add_1:z:0norm2/batchnorm/add_1:z:0concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ы2
concat§
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
ЫА*
dtype02
dense1/MatMul/ReadVariableOpТ
dense1/MatMulMatMulconcat:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/MatMulҐ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense1/BiasAdd/ReadVariableOpЮ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense1/Relu£
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense2/MatMul/ReadVariableOpЫ
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/MatMul°
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense2/BiasAdd/ReadVariableOpЭ
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense2/ReluҐ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
output/MatMul/ReadVariableOpЫ
output/MatMulMatMuldense2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/MatMul°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/BiasAddm
output/ReluReluoutput/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output/Relum
IdentityIdentityoutput/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ :::::::::::::::::::::::K G
(
_output_shapes
:€€€€€€€€€џ 
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
аn
й
 __inference__wrapped_model_86308
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
identityИk
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
model/split/split_dim∆
model/splitSplitVinput_1model/Const:output:0model/split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':€€€€€€€€€А :€€€€€€€€€[*
	num_split2
model/splitГ
model/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         2
model/Reshape/shapeЧ
model/ReshapeReshapemodel/split:output:0model/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
model/Reshapeє
!model/conv1/Conv2D/ReadVariableOpReadVariableOp*model_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!model/conv1/Conv2D/ReadVariableOpЎ
model/conv1/Conv2DConv2Dmodel/Reshape:output:0)model/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
model/conv1/Conv2D∞
"model/conv1/BiasAdd/ReadVariableOpReadVariableOp+model_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/conv1/BiasAdd/ReadVariableOpЄ
model/conv1/BiasAddBiasAddmodel/conv1/Conv2D:output:0*model/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
model/conv1/BiasAddД
model/conv1/ReluRelumodel/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
model/conv1/Reluє
!model/conv2/Conv2D/ReadVariableOpReadVariableOp*model_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!model/conv2/Conv2D/ReadVariableOpа
model/conv2/Conv2DConv2Dmodel/conv1/Relu:activations:0)model/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
model/conv2/Conv2D∞
"model/conv2/BiasAdd/ReadVariableOpReadVariableOp+model_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/conv2/BiasAdd/ReadVariableOpЄ
model/conv2/BiasAddBiasAddmodel/conv2/Conv2D:output:0*model/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
model/conv2/BiasAddД
model/conv2/ReluRelumodel/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
model/conv2/ReluЅ
model/pool1/MaxPoolMaxPoolmodel/conv2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingSAME*
strides
2
model/pool1/MaxPoolє
!model/conv3/Conv2D/ReadVariableOpReadVariableOp*model_conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!model/conv3/Conv2D/ReadVariableOpЁ
model/conv3/Conv2DConv2Dmodel/pool1/MaxPool:output:0)model/conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingSAME*
strides
2
model/conv3/Conv2D∞
"model/conv3/BiasAdd/ReadVariableOpReadVariableOp+model_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/conv3/BiasAdd/ReadVariableOpЄ
model/conv3/BiasAddBiasAddmodel/conv3/Conv2D:output:0*model/conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
model/conv3/BiasAddД
model/conv3/ReluRelumodel/conv3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
model/conv3/Reluє
!model/conv4/Conv2D/ReadVariableOpReadVariableOp*model_conv4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02#
!model/conv4/Conv2D/ReadVariableOpя
model/conv4/Conv2DConv2Dmodel/conv3/Relu:activations:0)model/conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
model/conv4/Conv2D∞
"model/conv4/BiasAdd/ReadVariableOpReadVariableOp+model_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/conv4/BiasAdd/ReadVariableOpЄ
model/conv4/BiasAddBiasAddmodel/conv4/Conv2D:output:0*model/conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
model/conv4/BiasAddД
model/conv4/ReluRelumodel/conv4/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
model/conv4/ReluЅ
model/pool2/MaxPoolMaxPoolmodel/conv4/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
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
valueB"€€€€@  2
model/flatten/Const®
model/flatten/ReshapeReshapemodel/pool2/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
model/flatten/Reshape®
model/norm1/Cast/ReadVariableOpReadVariableOp(model_norm1_cast_readvariableop_resource*
_output_shapes	
:ј*
dtype02!
model/norm1/Cast/ReadVariableOpЃ
!model/norm1/Cast_1/ReadVariableOpReadVariableOp*model_norm1_cast_1_readvariableop_resource*
_output_shapes	
:ј*
dtype02#
!model/norm1/Cast_1/ReadVariableOpЃ
!model/norm1/Cast_2/ReadVariableOpReadVariableOp*model_norm1_cast_2_readvariableop_resource*
_output_shapes	
:ј*
dtype02#
!model/norm1/Cast_2/ReadVariableOpЃ
!model/norm1/Cast_3/ReadVariableOpReadVariableOp*model_norm1_cast_3_readvariableop_resource*
_output_shapes	
:ј*
dtype02#
!model/norm1/Cast_3/ReadVariableOp
model/norm1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
model/norm1/batchnorm/add/yґ
model/norm1/batchnorm/addAddV2)model/norm1/Cast_1/ReadVariableOp:value:0$model/norm1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ј2
model/norm1/batchnorm/addИ
model/norm1/batchnorm/RsqrtRsqrtmodel/norm1/batchnorm/add:z:0*
T0*
_output_shapes	
:ј2
model/norm1/batchnorm/Rsqrtѓ
model/norm1/batchnorm/mulMulmodel/norm1/batchnorm/Rsqrt:y:0)model/norm1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ј2
model/norm1/batchnorm/mul≥
model/norm1/batchnorm/mul_1Mulmodel/flatten/Reshape:output:0model/norm1/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
model/norm1/batchnorm/mul_1ѓ
model/norm1/batchnorm/mul_2Mul'model/norm1/Cast/ReadVariableOp:value:0model/norm1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ј2
model/norm1/batchnorm/mul_2ѓ
model/norm1/batchnorm/subSub)model/norm1/Cast_2/ReadVariableOp:value:0model/norm1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ј2
model/norm1/batchnorm/subґ
model/norm1/batchnorm/add_1AddV2model/norm1/batchnorm/mul_1:z:0model/norm1/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
model/norm1/batchnorm/add_1І
model/norm2/Cast/ReadVariableOpReadVariableOp(model_norm2_cast_readvariableop_resource*
_output_shapes
:[*
dtype02!
model/norm2/Cast/ReadVariableOp≠
!model/norm2/Cast_1/ReadVariableOpReadVariableOp*model_norm2_cast_1_readvariableop_resource*
_output_shapes
:[*
dtype02#
!model/norm2/Cast_1/ReadVariableOp≠
!model/norm2/Cast_2/ReadVariableOpReadVariableOp*model_norm2_cast_2_readvariableop_resource*
_output_shapes
:[*
dtype02#
!model/norm2/Cast_2/ReadVariableOp≠
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
 *oГ:2
model/norm2/batchnorm/add/yµ
model/norm2/batchnorm/addAddV2)model/norm2/Cast_1/ReadVariableOp:value:0$model/norm2/batchnorm/add/y:output:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/addЗ
model/norm2/batchnorm/RsqrtRsqrtmodel/norm2/batchnorm/add:z:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/RsqrtЃ
model/norm2/batchnorm/mulMulmodel/norm2/batchnorm/Rsqrt:y:0)model/norm2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/mul®
model/norm2/batchnorm/mul_1Mulmodel/split:output:1model/norm2/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
model/norm2/batchnorm/mul_1Ѓ
model/norm2/batchnorm/mul_2Mul'model/norm2/Cast/ReadVariableOp:value:0model/norm2/batchnorm/mul:z:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/mul_2Ѓ
model/norm2/batchnorm/subSub)model/norm2/Cast_2/ReadVariableOp:value:0model/norm2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:[2
model/norm2/batchnorm/subµ
model/norm2/batchnorm/add_1AddV2model/norm2/batchnorm/mul_1:z:0model/norm2/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
model/norm2/batchnorm/add_1h
model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concat/axis¬
model/concatConcatV2model/norm1/batchnorm/add_1:z:0model/norm2/batchnorm/add_1:z:0model/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ы2
model/concatґ
"model/dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource* 
_output_shapes
:
ЫА*
dtype02$
"model/dense1/MatMul/ReadVariableOp™
model/dense1/MatMulMatMulmodel/concat:output:0*model/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dense1/MatMulі
#model/dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#model/dense1/BiasAdd/ReadVariableOpґ
model/dense1/BiasAddBiasAddmodel/dense1/MatMul:product:0+model/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dense1/BiasAddА
model/dense1/ReluRelumodel/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model/dense1/Reluµ
"model/dense2/MatMul/ReadVariableOpReadVariableOp+model_dense2_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02$
"model/dense2/MatMul/ReadVariableOp≥
model/dense2/MatMulMatMulmodel/dense1/Relu:activations:0*model/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model/dense2/MatMul≥
#model/dense2/BiasAdd/ReadVariableOpReadVariableOp,model_dense2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/dense2/BiasAdd/ReadVariableOpµ
model/dense2/BiasAddBiasAddmodel/dense2/MatMul:product:0+model/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model/dense2/BiasAdd
model/dense2/ReluRelumodel/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model/dense2/Reluі
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02$
"model/output/MatMul/ReadVariableOp≥
model/output/MatMulMatMulmodel/dense2/Relu:activations:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/output/MatMul≥
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/output/BiasAdd/ReadVariableOpµ
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/output/BiasAdd
model/output/ReluRelumodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/output/Relus
IdentityIdentitymodel/output/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ :::::::::::::::::::::::Q M
(
_output_shapes
:€€€€€€€€€џ 
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
ў
z
%__inference_conv1_layer_call_fn_86331

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_863212
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
”
ђ
%__inference_model_layer_call_fn_87786
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
identityИҐStatefulPartitionedCallџ
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
:€€€€€€€€€*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_870402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:€€€€€€€€€џ 
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
“)
Ђ
@__inference_norm1_layer_call_and_return_conditional_losses_86517

inputs
assignmovingavg_86492
assignmovingavg_1_86498 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ј*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ј2
moments/StopGradient•
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≥
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ј*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ј*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ј*
squeeze_dims
 2
moments/Squeeze_1Э
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/86492*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_86492*
_output_shapes	
:ј*
dtype02 
AssignMovingAvg/ReadVariableOp√
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/86492*
_output_shapes	
:ј2
AssignMovingAvg/subЇ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/86492*
_output_shapes	
:ј2
AssignMovingAvg/mul€
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_86492AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/86492*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp£
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/86498*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_86498*
_output_shapes	
:ј*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/86498*
_output_shapes	
:ј2
AssignMovingAvg_1/subƒ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/86498*
_output_shapes	
:ј2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_86498AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/86498*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpД
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ј2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ј2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ј2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ј2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ј2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
batchnorm/add_1ґ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ј::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј
 
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
е
©
A__inference_dense2_layer_call_and_return_conditional_losses_87877

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ш
A
%__inference_pool2_layer_call_fn_86421

inputs
identity¬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_pool2_layer_call_and_return_conditional_losses_864152
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
у
{
&__inference_output_layer_call_fn_87906

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_868842
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
л
©
A__inference_dense1_layer_call_and_return_conditional_losses_87857

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЫА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Ы:::P L
(
_output_shapes
:€€€€€€€€€Ы
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ї?
щ
@__inference_model_layer_call_and_return_conditional_losses_87040
x
conv1_86981
conv1_86983
conv2_86986
conv2_86988
conv3_86992
conv3_86994
conv4_86997
conv4_86999
norm1_87004
norm1_87006
norm1_87008
norm1_87010
norm2_87013
norm2_87015
norm2_87017
norm2_87019
dense1_87024
dense1_87026
dense2_87029
dense2_87031
output_87034
output_87036
identityИҐconv1/StatefulPartitionedCallҐconv2/StatefulPartitionedCallҐconv3/StatefulPartitionedCallҐconv4/StatefulPartitionedCallҐdense1/StatefulPartitionedCallҐdense2/StatefulPartitionedCallҐnorm1/StatefulPartitionedCallҐnorm2/StatefulPartitionedCallҐoutput/StatefulPartitionedCall_
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
split/split_dim®
splitSplitVxConst:output:0split/split_dim:output:0*
T0*

Tlen0*;
_output_shapes)
':€€€€€€€€€А :€€€€€€€€€[*
	num_split2
splitw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         2
Reshape/shape
ReshapeReshapesplit:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2	
Reshapeх
conv1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1_86981conv1_86983*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_863212
conv1/StatefulPartitionedCallЛ
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_86986conv2_86988*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_863432
conv2/StatefulPartitionedCall”
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_pool1_layer_call_and_return_conditional_losses_863592
pool1/PartitionedCallГ
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv3_86992conv3_86994*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_863772
conv3/StatefulPartitionedCallЛ
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0conv4_86997conv4_86999*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_863992
conv4/StatefulPartitionedCall”
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_pool2_layer_call_and_return_conditional_losses_864152
pool2/PartitionedCall 
flatten/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_867392
flatten/PartitionedCallЬ
norm1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0norm1_87004norm1_87006norm1_87008norm1_87010*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€ј*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_norm1_layer_call_and_return_conditional_losses_865502
norm1/StatefulPartitionedCallЙ
norm2/StatefulPartitionedCallStatefulPartitionedCallsplit:output:1norm2_87013norm2_87015norm2_87017norm2_87019*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€[*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_norm2_layer_call_and_return_conditional_losses_866902
norm2/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЊ
concatConcatV2&norm1/StatefulPartitionedCall:output:0&norm2/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ы2
concatт
dense1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense1_87024dense1_87026*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_868302 
dense1/StatefulPartitionedCallЙ
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_87029dense2_87031*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_868572 
dense2/StatefulPartitionedCallЙ
output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_87034output_87036*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_868842 
output/StatefulPartitionedCallЮ
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^norm1/StatefulPartitionedCall^norm2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ ::::::::::::::::::::::2>
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
:€€€€€€€€€џ 
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
Ї
^
B__inference_flatten_layer_call_and_return_conditional_losses_86739

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ї)
Ђ
@__inference_norm2_layer_call_and_return_conditional_losses_88024

inputs
assignmovingavg_87999
assignmovingavg_1_88005 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
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
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€[2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
moments/Squeeze_1Э
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/87999*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_87999*
_output_shapes
:[*
dtype02 
AssignMovingAvg/ReadVariableOp¬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/87999*
_output_shapes
:[2
AssignMovingAvg/subє
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/87999*
_output_shapes
:[2
AssignMovingAvg/mul€
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_87999AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/87999*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp£
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/88005*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_88005*
_output_shapes
:[*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/88005*
_output_shapes
:[2
AssignMovingAvg_1/sub√
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/88005*
_output_shapes
:[2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_88005AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/88005*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
:€€€€€€€€€[2
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
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:€€€€€€€€€[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€[::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:€€€€€€€€€[
 
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
Ѓ

®
@__inference_conv3_layer_call_and_return_conditional_losses_86377

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
х
\
@__inference_pool2_layer_call_and_return_conditional_losses_86415

inputs
identityђ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
и:
с	
__inference__traced_save_88163
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

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1П
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_293168b39d9046c4b5a1b3b61279f2d8/part2	
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЧ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueЯBЬB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d3/bias/.ATTRIBUTES/VARIABLE_VALUEB)b_norm_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB(b_norm_1/beta/.ATTRIBUTES/VARIABLE_VALUEB/b_norm_1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3b_norm_1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)b_norm_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB(b_norm_2/beta/.ATTRIBUTES/VARIABLE_VALUEB/b_norm_2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3b_norm_2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesі
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesў	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_model_conv1_kernel_read_readvariableop+savev2_model_conv1_bias_read_readvariableop-savev2_model_conv2_kernel_read_readvariableop+savev2_model_conv2_bias_read_readvariableop-savev2_model_conv3_kernel_read_readvariableop+savev2_model_conv3_bias_read_readvariableop-savev2_model_conv4_kernel_read_readvariableop+savev2_model_conv4_bias_read_readvariableop.savev2_model_dense1_kernel_read_readvariableop,savev2_model_dense1_bias_read_readvariableop.savev2_model_dense2_kernel_read_readvariableop,savev2_model_dense2_bias_read_readvariableop.savev2_model_output_kernel_read_readvariableop,savev2_model_output_bias_read_readvariableop,savev2_model_norm1_gamma_read_readvariableop+savev2_model_norm1_beta_read_readvariableop2savev2_model_norm1_moving_mean_read_readvariableop6savev2_model_norm1_moving_variance_read_readvariableop,savev2_model_norm2_gamma_read_readvariableop+savev2_model_norm2_beta_read_readvariableop2savev2_model_norm2_moving_mean_read_readvariableop6savev2_model_norm2_moving_variance_read_readvariableop"/device:CPU:0*
_output_shapes
 *$
dtypes
22
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*б
_input_shapesѕ
ћ: ::: : :  : : @:@:
ЫА:А:	А@:@:@::ј:ј:ј:ј:[:[:[:[: 2(
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
ЫА:!


_output_shapes	
:А:%!

_output_shapes
:	А@: 
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
:ј:!

_output_shapes	
:ј:!

_output_shapes	
:ј:!

_output_shapes	
:ј: 
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
щ
н
@__inference_norm2_layer_call_and_return_conditional_losses_88044

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityИГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
:€€€€€€€€€[2
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
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€[:::::O K
'
_output_shapes
:€€€€€€€€€[
 
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
в
©
A__inference_output_layer_call_and_return_conditional_losses_87897

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:::O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
√
∞
#__inference_signature_wrapper_87187
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
identityИҐStatefulPartitionedCallЅ
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
:€€€€€€€€€*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__wrapped_model_863082
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€џ 
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
≤
%__inference_model_layer_call_fn_87511
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
identityИҐStatefulPartitionedCallб
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
:€€€€€€€€€*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_870402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€џ ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€џ 
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
ѓ

®
@__inference_conv2_layer_call_and_return_conditional_losses_86343

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ў
z
%__inference_conv3_layer_call_fn_86387

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_863772
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
е
©
A__inference_dense2_layer_call_and_return_conditional_losses_86857

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
“)
Ђ
@__inference_norm1_layer_call_and_return_conditional_losses_87942

inputs
assignmovingavg_87917
assignmovingavg_1_87923 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ј*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ј2
moments/StopGradient•
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≥
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ј*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ј*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ј*
squeeze_dims
 2
moments/Squeeze_1Э
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/87917*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_87917*
_output_shapes	
:ј*
dtype02 
AssignMovingAvg/ReadVariableOp√
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/87917*
_output_shapes	
:ј2
AssignMovingAvg/subЇ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/87917*
_output_shapes	
:ј2
AssignMovingAvg/mul€
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_87917AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/87917*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp£
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/87923*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_87923*
_output_shapes	
:ј*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/87923*
_output_shapes	
:ј2
AssignMovingAvg_1/subƒ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/87923*
_output_shapes	
:ј2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_87923AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/87923*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpД
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ј*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ј2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ј2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ј2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ј2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ј2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
batchnorm/add_1ґ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ј::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј
 
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
Ї)
Ђ
@__inference_norm2_layer_call_and_return_conditional_losses_86657

inputs
assignmovingavg_86632
assignmovingavg_1_86638 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
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
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€[2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:[*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:[*
squeeze_dims
 2
moments/Squeeze_1Э
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/86632*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_86632*
_output_shapes
:[*
dtype02 
AssignMovingAvg/ReadVariableOp¬
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/86632*
_output_shapes
:[2
AssignMovingAvg/subє
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/86632*
_output_shapes
:[2
AssignMovingAvg/mul€
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_86632AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/86632*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp£
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/86638*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_86638*
_output_shapes
:[*
dtype02"
 AssignMovingAvg_1/ReadVariableOpћ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/86638*
_output_shapes
:[2
AssignMovingAvg_1/sub√
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/86638*
_output_shapes
:[2
AssignMovingAvg_1/mulЛ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_86638AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/86638*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
:€€€€€€€€€[2
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
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:€€€€€€€€€[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€[::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:€€€€€€€€€[
 
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
о
Ш
%__inference_norm2_layer_call_fn_88070

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€[*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_norm2_layer_call_and_return_conditional_losses_866902
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€[::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€[
 
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
™`
Д
!__inference__traced_restore_88241
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
identity_23ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1Э
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueЯBЬB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv4/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB$d3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d3/bias/.ATTRIBUTES/VARIABLE_VALUEB)b_norm_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB(b_norm_1/beta/.ATTRIBUTES/VARIABLE_VALUEB/b_norm_1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3b_norm_1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)b_norm_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB(b_norm_2/beta/.ATTRIBUTES/VARIABLE_VALUEB/b_norm_2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB3b_norm_2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЇ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЩ
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

IdentityУ
AssignVariableOpAssignVariableOp#assignvariableop_model_conv1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Щ
AssignVariableOp_1AssignVariableOp#assignvariableop_1_model_conv1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ы
AssignVariableOp_2AssignVariableOp%assignvariableop_2_model_conv2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Щ
AssignVariableOp_3AssignVariableOp#assignvariableop_3_model_conv2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ы
AssignVariableOp_4AssignVariableOp%assignvariableop_4_model_conv3_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Щ
AssignVariableOp_5AssignVariableOp#assignvariableop_5_model_conv3_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ы
AssignVariableOp_6AssignVariableOp%assignvariableop_6_model_conv4_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Щ
AssignVariableOp_7AssignVariableOp#assignvariableop_7_model_conv4_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ь
AssignVariableOp_8AssignVariableOp&assignvariableop_8_model_dense1_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ъ
AssignVariableOp_9AssignVariableOp$assignvariableop_9_model_dense1_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10†
AssignVariableOp_10AssignVariableOp'assignvariableop_10_model_dense2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ю
AssignVariableOp_11AssignVariableOp%assignvariableop_11_model_dense2_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12†
AssignVariableOp_12AssignVariableOp'assignvariableop_12_model_output_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ю
AssignVariableOp_13AssignVariableOp%assignvariableop_13_model_output_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ю
AssignVariableOp_14AssignVariableOp%assignvariableop_14_model_norm1_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Э
AssignVariableOp_15AssignVariableOp$assignvariableop_15_model_norm1_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16§
AssignVariableOp_16AssignVariableOp+assignvariableop_16_model_norm1_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17®
AssignVariableOp_17AssignVariableOp/assignvariableop_17_model_norm1_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ю
AssignVariableOp_18AssignVariableOp%assignvariableop_18_model_norm2_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Э
AssignVariableOp_19AssignVariableOp$assignvariableop_19_model_norm2_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20§
AssignVariableOp_20AssignVariableOp+assignvariableop_20_model_norm2_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp/assignvariableop_21_model_norm2_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
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
NoOp¬
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22ѕ
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
ў
z
%__inference_conv2_layer_call_fn_86353

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_863432
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ
н
@__inference_norm2_layer_call_and_return_conditional_losses_86690

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identityИГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:[*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
:€€€€€€€€€[2
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
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€[2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€[:::::O K
'
_output_shapes
:€€€€€€€€€[
 
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
х
{
&__inference_dense2_layer_call_fn_87886

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_868572
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
т
Ш
%__inference_norm1_layer_call_fn_87988

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€ј*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_norm1_layer_call_and_return_conditional_losses_865502
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ј::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ј
 
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
: "ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ђ
serving_defaultШ
<
input_11
serving_default_input_1:0€€€€€€€€€џ <
output_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:€Ч
У
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
Ы_default_save_signature
Ь__call__
+Э&call_and_return_all_conditional_losses"—
_tf_keras_modelЈ{"class_name": "model", "name": "model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "model"}}
љ	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"Ц
_tf_keras_layerь{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 16, 16, 16]}}
љ	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
†__call__
+°&call_and_return_all_conditional_losses"Ц
_tf_keras_layerь{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 14, 14, 16]}}
…
trainable_variables
regularization_losses
 	variables
!	keras_api
Ґ__call__
+£&call_and_return_all_conditional_losses"Є
_tf_keras_layerЮ{"class_name": "MaxPooling2D", "name": "pool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ї	

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
§__call__
+•&call_and_return_all_conditional_losses"У
_tf_keras_layerщ{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 6, 6, 32]}}
Ї	

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
¶__call__
+І&call_and_return_all_conditional_losses"У
_tf_keras_layerщ{"class_name": "Conv2D", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 6, 6, 32]}}
…
.trainable_variables
/regularization_losses
0	variables
1	keras_api
®__call__
+©&call_and_return_all_conditional_losses"Є
_tf_keras_layerЮ{"class_name": "MaxPooling2D", "name": "pool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ѕ
2trainable_variables
3regularization_losses
4	variables
5	keras_api
™__call__
+Ђ&call_and_return_all_conditional_losses"∞
_tf_keras_layerЦ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ќ

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses"І
_tf_keras_layerН{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 667}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 667]}}
Ќ

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses"¶
_tf_keras_layerМ{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 128]}}
 

Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
∞__call__
+±&call_and_return_all_conditional_losses"£
_tf_keras_layerЙ{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 64]}}
с
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "BatchNormalization", "name": "norm1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "norm1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 576}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 576]}}
п
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
і__call__
+µ&call_and_return_all_conditional_losses"Щ
_tf_keras_layer€{"class_name": "BatchNormalization", "name": "norm2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "norm2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 91}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 91]}}
¶
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
 "
trackable_list_wrapper
∆
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
ќ
Znon_trainable_variables
[layer_metrics
trainable_variables
regularization_losses

\layers
]metrics
^layer_regularization_losses
	variables
Ь__call__
Ы_default_save_signature
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
-
ґserving_default"
signature_map
,:*2model/conv1/kernel
:2model/conv1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
_non_trainable_variables
`layer_regularization_losses
trainable_variables
regularization_losses

alayers
bmetrics
clayer_metrics
	variables
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
,:* 2model/conv2/kernel
: 2model/conv2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
dnon_trainable_variables
elayer_regularization_losses
trainable_variables
regularization_losses

flayers
gmetrics
hlayer_metrics
	variables
†__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
inon_trainable_variables
jlayer_regularization_losses
trainable_variables
regularization_losses

klayers
lmetrics
mlayer_metrics
 	variables
Ґ__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
,:*  2model/conv3/kernel
: 2model/conv3/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
∞
nnon_trainable_variables
olayer_regularization_losses
$trainable_variables
%regularization_losses

players
qmetrics
rlayer_metrics
&	variables
§__call__
+•&call_and_return_all_conditional_losses
'•"call_and_return_conditional_losses"
_generic_user_object
,:* @2model/conv4/kernel
:@2model/conv4/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
∞
snon_trainable_variables
tlayer_regularization_losses
*trainable_variables
+regularization_losses

ulayers
vmetrics
wlayer_metrics
,	variables
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
xnon_trainable_variables
ylayer_regularization_losses
.trainable_variables
/regularization_losses

zlayers
{metrics
|layer_metrics
0	variables
®__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
}non_trainable_variables
~layer_regularization_losses
2trainable_variables
3regularization_losses

layers
Аmetrics
Бlayer_metrics
4	variables
™__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
':%
ЫА2model/dense1/kernel
 :А2model/dense1/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
µ
Вnon_trainable_variables
 Гlayer_regularization_losses
8trainable_variables
9regularization_losses
Дlayers
Еmetrics
Жlayer_metrics
:	variables
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
&:$	А@2model/dense2/kernel
:@2model/dense2/bias
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
µ
Зnon_trainable_variables
 Иlayer_regularization_losses
>trainable_variables
?regularization_losses
Йlayers
Кmetrics
Лlayer_metrics
@	variables
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
%:#@2model/output/kernel
:2model/output/bias
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
µ
Мnon_trainable_variables
 Нlayer_regularization_losses
Dtrainable_variables
Eregularization_losses
Оlayers
Пmetrics
Рlayer_metrics
F	variables
∞__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :ј2model/norm1/gamma
:ј2model/norm1/beta
(:&ј (2model/norm1/moving_mean
,:*ј (2model/norm1/moving_variance
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
I0
J1
K2
L3"
trackable_list_wrapper
µ
Сnon_trainable_variables
 Тlayer_regularization_losses
Mtrainable_variables
Nregularization_losses
Уlayers
Фmetrics
Хlayer_metrics
O	variables
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:[2model/norm2/gamma
:[2model/norm2/beta
':%[ (2model/norm2/moving_mean
+:)[ (2model/norm2/moving_variance
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
R0
S1
T2
U3"
trackable_list_wrapper
µ
Цnon_trainable_variables
 Чlayer_regularization_losses
Vtrainable_variables
Wregularization_losses
Шlayers
Щmetrics
Ъlayer_metrics
X	variables
і__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
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
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
я2№
 __inference__wrapped_model_86308Ј
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *'Ґ$
"К
input_1€€€€€€€€€џ 
–2Ќ
%__inference_model_layer_call_fn_87511
%__inference_model_layer_call_fn_87835
%__inference_model_layer_call_fn_87462
%__inference_model_layer_call_fn_87786Ѓ
•≤°
FullArgSpec$
argsЪ
jself
jX

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Љ2є
@__inference_model_layer_call_and_return_conditional_losses_87737
@__inference_model_layer_call_and_return_conditional_losses_87316
@__inference_model_layer_call_and_return_conditional_losses_87413
@__inference_model_layer_call_and_return_conditional_losses_87640Ѓ
•≤°
FullArgSpec$
argsЪ
jself
jX

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Д2Б
%__inference_conv1_layer_call_fn_86331„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Я2Ь
@__inference_conv1_layer_call_and_return_conditional_losses_86321„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Д2Б
%__inference_conv2_layer_call_fn_86353„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Я2Ь
@__inference_conv2_layer_call_and_return_conditional_losses_86343„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Н2К
%__inference_pool1_layer_call_fn_86365а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®2•
@__inference_pool1_layer_call_and_return_conditional_losses_86359а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Д2Б
%__inference_conv3_layer_call_fn_86387„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Я2Ь
@__inference_conv3_layer_call_and_return_conditional_losses_86377„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Д2Б
%__inference_conv4_layer_call_fn_86409„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Я2Ь
@__inference_conv4_layer_call_and_return_conditional_losses_86399„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Н2К
%__inference_pool2_layer_call_fn_86421а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®2•
@__inference_pool2_layer_call_and_return_conditional_losses_86415а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
—2ќ
'__inference_flatten_layer_call_fn_87846Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_flatten_layer_call_and_return_conditional_losses_87841Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_dense1_layer_call_fn_87866Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_dense1_layer_call_and_return_conditional_losses_87857Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_dense2_layer_call_fn_87886Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_dense2_layer_call_and_return_conditional_losses_87877Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_output_layer_call_fn_87906Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_output_layer_call_and_return_conditional_losses_87897Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
И2Е
%__inference_norm1_layer_call_fn_87975
%__inference_norm1_layer_call_fn_87988і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Њ2ї
@__inference_norm1_layer_call_and_return_conditional_losses_87942
@__inference_norm1_layer_call_and_return_conditional_losses_87962і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
И2Е
%__inference_norm2_layer_call_fn_88070
%__inference_norm2_layer_call_fn_88057і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Њ2ї
@__inference_norm2_layer_call_and_return_conditional_losses_88024
@__inference_norm2_layer_call_and_return_conditional_losses_88044і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
2B0
#__inference_signature_wrapper_87187input_1•
 __inference__wrapped_model_86308А"#()KLJITUSR67<=BC1Ґ.
'Ґ$
"К
input_1€€€€€€€€€џ 
™ "3™0
.
output_1"К
output_1€€€€€€€€€’
@__inference_conv1_layer_call_and_return_conditional_losses_86321РIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≠
%__inference_conv1_layer_call_fn_86331ГIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€’
@__inference_conv2_layer_call_and_return_conditional_losses_86343РIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ≠
%__inference_conv2_layer_call_fn_86353ГIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ’
@__inference_conv3_layer_call_and_return_conditional_losses_86377Р"#IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ≠
%__inference_conv3_layer_call_fn_86387Г"#IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ’
@__inference_conv4_layer_call_and_return_conditional_losses_86399Р()IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≠
%__inference_conv4_layer_call_fn_86409Г()IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@£
A__inference_dense1_layer_call_and_return_conditional_losses_87857^670Ґ-
&Ґ#
!К
inputs€€€€€€€€€Ы
™ "&Ґ#
К
0€€€€€€€€€А
Ъ {
&__inference_dense1_layer_call_fn_87866Q670Ґ-
&Ґ#
!К
inputs€€€€€€€€€Ы
™ "К€€€€€€€€€АҐ
A__inference_dense2_layer_call_and_return_conditional_losses_87877]<=0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ z
&__inference_dense2_layer_call_fn_87886P<=0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€@І
B__inference_flatten_layer_call_and_return_conditional_losses_87841a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ 
'__inference_flatten_layer_call_fn_87846T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ "К€€€€€€€€€јЇ
@__inference_model_layer_call_and_return_conditional_losses_87316v"#()KLJITUSR67<=BC5Ґ2
+Ґ(
"К
input_1€€€€€€€€€џ 
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ї
@__inference_model_layer_call_and_return_conditional_losses_87413v"#()KLJITUSR67<=BC5Ґ2
+Ґ(
"К
input_1€€€€€€€€€џ 
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ і
@__inference_model_layer_call_and_return_conditional_losses_87640p"#()KLJITUSR67<=BC/Ґ,
%Ґ"
К
X€€€€€€€€€џ 
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ і
@__inference_model_layer_call_and_return_conditional_losses_87737p"#()KLJITUSR67<=BC/Ґ,
%Ґ"
К
X€€€€€€€€€џ 
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Т
%__inference_model_layer_call_fn_87462i"#()KLJITUSR67<=BC5Ґ2
+Ґ(
"К
input_1€€€€€€€€€џ 
p
™ "К€€€€€€€€€Т
%__inference_model_layer_call_fn_87511i"#()KLJITUSR67<=BC5Ґ2
+Ґ(
"К
input_1€€€€€€€€€џ 
p 
™ "К€€€€€€€€€М
%__inference_model_layer_call_fn_87786c"#()KLJITUSR67<=BC/Ґ,
%Ґ"
К
X€€€€€€€€€џ 
p
™ "К€€€€€€€€€М
%__inference_model_layer_call_fn_87835c"#()KLJITUSR67<=BC/Ґ,
%Ґ"
К
X€€€€€€€€€џ 
p 
™ "К€€€€€€€€€®
@__inference_norm1_layer_call_and_return_conditional_losses_87942dKLJI4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ ®
@__inference_norm1_layer_call_and_return_conditional_losses_87962dKLJI4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p 
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ А
%__inference_norm1_layer_call_fn_87975WKLJI4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p
™ "К€€€€€€€€€јА
%__inference_norm1_layer_call_fn_87988WKLJI4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p 
™ "К€€€€€€€€€ј¶
@__inference_norm2_layer_call_and_return_conditional_losses_88024bTUSR3Ґ0
)Ґ&
 К
inputs€€€€€€€€€[
p
™ "%Ґ"
К
0€€€€€€€€€[
Ъ ¶
@__inference_norm2_layer_call_and_return_conditional_losses_88044bTUSR3Ґ0
)Ґ&
 К
inputs€€€€€€€€€[
p 
™ "%Ґ"
К
0€€€€€€€€€[
Ъ ~
%__inference_norm2_layer_call_fn_88057UTUSR3Ґ0
)Ґ&
 К
inputs€€€€€€€€€[
p
™ "К€€€€€€€€€[~
%__inference_norm2_layer_call_fn_88070UTUSR3Ґ0
)Ґ&
 К
inputs€€€€€€€€€[
p 
™ "К€€€€€€€€€[°
A__inference_output_layer_call_and_return_conditional_losses_87897\BC/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ y
&__inference_output_layer_call_fn_87906OBC/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€г
@__inference_pool1_layer_call_and_return_conditional_losses_86359ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ї
%__inference_pool1_layer_call_fn_86365СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€г
@__inference_pool2_layer_call_and_return_conditional_losses_86415ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ї
%__inference_pool2_layer_call_fn_86421СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
#__inference_signature_wrapper_87187Л"#()KLJITUSR67<=BC<Ґ9
Ґ 
2™/
-
input_1"К
input_1€€€€€€€€€џ "3™0
.
output_1"К
output_1€€€€€€€€€