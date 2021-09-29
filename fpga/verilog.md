## Verilog简介

### 1. Verilog设计

> 设计建模：
>
> - 行为级描述——使用过程化结构建模；
> - 数据流描述——使用连续赋值语句建模；
> - 结构化方式——使用门和模块例化语句描述。
>
> 两类数据类型：
>
> - 线网（wire）数据类型，线网表示物理元件之间的连线。
> - 寄存器（reg）数据类型，寄存器表示抽象的数据存储元件。

Verilog 的设计多采用自上而下的设计方法（top-down）。即先定义顶层模块功能，进而分析要构成顶层模块的必要子模块；然后进一步对各个模块进行分解、设计，直到到达无法进一步分解的底层功能块。

### 2. 基本语法

- 除了endmodule，每个语句和数据定义必须以分号为结束符。
- 用 **//** 进行单行注释；用 **/\*** 与 ***/** 进行跨行注释。
- 标识符（identifier）可以是任意一组字母、数字、**$** 符号和 **_**(下划线)符号的合，但标识符的第一个字符必须是字母或者下划线，不能以数字或者美元符开始。
- 一行可以写几个语句，一个语句也可以分写多行。
- 模块的端口定义：module 模块名(口1，口2，口3.......);

### 3. 基本构成

- Verilog程序是由模块构成，每个模块的内容都是嵌在`module`和`endmodule`两个语句之间。模块是可以进行层次嵌套的。正因为如此，才可以将大型的数字电路设计分割成不同的小模块来实现特定的功能，最后通过顶层模块调用子模块来实现整体功能。

- 每个模块要**进行端口定义，并说明输入输出口**，然后对模块的功能进行行为逻辑描述。

  ```verilog
  module 模块名(口1，口2，口3.......);
   	input ;
    output ;
    ......
  endmodule
  ```

- 每个模块构成：

  - IO说明：

    ```verilog
    input  口1,口2......;    
    output  口1,口2......;
    ```

  - 内部信号说明:

    ```verilog
    reg[width-1:0] R1,R2;
    wire[width-1:0] W1,W2; 
    ```

  -  功能定义：

    > **如果用Verilog模块实现一定的功能，首先应该清楚哪些是同时发生的，哪些是顺序发生的。**
    >
    > - 下面三个例子分别采用了“assign”语句、实例元件和“always”块。这三个例子描述的逻辑功能是**同时执行的**。也就是说，如果把这三项写到一个 VeriIog 模块文件中去，它们的次序不会影响逻辑实现的功能。**这三项是同时执行的，也就是并发的**。 
    >
    > - 在“always”模块内，**逻辑是按照指定的顺序执行的**。“always”块中的语句称为“顺序语句”，因为它们是顺序执行的。请注意，**两个或更多的“always”模块也是同时执行的**，但是模块内部的语句是顺序执行的。看一下“always”内的语句，你就会明白它是如何实现功能的。if..else… if必须顺序执行，否则其功能就没有任何意义。如果else语句在if语句之前执行，功能就会不符合要求。为了能实现上述描述的功能，“always”模块内部的语句将按照书写的顺序执行。

    ```verilog
    //1.组合逻辑
    assign a = b&c; 
    //2.时序逻辑或者组合逻辑
    always @(posedge clk or posedge clr)
      begin
        if(clr) q<=0;
        else if(en) q<=d;
      end
    //3.实例元件
    and and_inst(q,a,b);
    ```

## 数据类型

>数据类型是用来表示数字电路中的数据存储形式和传送单元。
>
>Verilog HDL中总共有十九种数据类型：reg型、wire型、integer型、parameter型、large型、medium型、scalared型、time型、small型、tri型、trio型、tri1型、triand型、trior型、trireg型、vectored型、wand型、wor型。这些数据类型除time型外都与基本逻辑单元建库有关，与系统设计没有很大的关系。

### 1. 常量

#### （1）数值

- 电平逻辑
  - 0：逻辑 0 或 "假"
  - 1：逻辑 1 或 "真"
  - x 或 X：未知
  - z 或 Z：高阻

> **x** 意味着信号数值的不确定，即在实际电路里，信号可能为 1，也可能为 0。
>
> **z** 意味着信号处于高阻状态，常见于信号（input, reg）没有驱动时的逻辑结果。例如一个 pad 的 input 呈现高阻状态时，其逻辑值和上下拉的状态有关系。上拉则逻辑值为 1，下拉则为 0 。
>
> **已标明位宽的数若用x或z表示某些位，则只有在最左边的x或z具有扩展性!为清晰可见，最好直接写出每一位的值！** 8’bzx = 8’bzzzz_zzzx		8’b1x = 8’b0000_001x 
>
> ?是z的另一种表示符号，建议在case语句中使用?表示高阻态z
>
> 一个x可以用来定义十六进制数的四位二进制数的状态,八进制数的三位,二进制数的一位。z的表示方式同x类似。

- 整数
  - <位宽> ' <进制><数字>	
  -  ' <进制><数字>   一般会根据编译器自动分频位宽，常见的为32bit
  - <数字>  缺省进制十进制，位宽默认32位

> 基数格式有 4 中，包括：十进制('d 或 'D)，十六进制('h 或 'H)，二进制（'b 或 'B），八进制（'o 或 'O）。
>

```verilog
4'b1011         // 4bit 数值
32'h3022_c0de   // 32bit 的数值,下划线 _ 是为了增强代码的可读性。
counter = 'd100 ; //一般会根据编译器自动分频位宽，常见的为32bit
counter = 100 ;
counter = 32'h64 ;
-6'd15  //通常在表示位宽的数字前面加一个减号来表示负数
```

- 实数

```verilog
//十进制
30.123
//科学计数
1.2e4         //大小为12000
1_0001e4      //大小为100010000
1E-3          //大小为0.001
```

#### （2）字符串

> 字符串是由双引号包起来的字符队列。字符串中不能包含回车符。Verilog 将字符串当做一系列的单字节 ASCII 字符队列。例如，为存储字符串 "www.runoob. com", 需要 14*8bit 的存储单元。

```verilog
reg [0: 14*8-1]       str ;
assign str = "www.runoob.com";
```

#### （3）parameter

> 参数用来表示常量，用关键字 parameter 声明，**只能赋值一次。**
>
> **通过实例化的方式，可以更改参数在模块中的值，两种方法**。
>
> 局部参数用 localparam 来声明，其作用和用法与 parameter 相同，区别在于它的值不能被改变。
>
> 参数类型常常用于定义延迟时间和变量宽度

```verilog
parameter  data_width = 10'd32 ;

module mod ( out, ina, inb); 
  parameter cycle = 8, real_constant = 2.039, file = “/design/mem_file.dat”; 
endmodule
//第一种修改方法defparam
module test;
  mod mk(out,ina,inb); // 对模块mod的实例引用
  defparam mk.cycle = 6, mk.file = “../my_mem.dat”; // 参数的传递
endmodule
//第二种修改方法#
module test1;
  mod # (5, 3.20, “../my_mem.dat”) mk(out,ina,inb);// 对模块mod的实例引
endmodule
```

### 2. 变量

#### （1）wire

> 表示硬件单元之间的物理连线，由其连接的器件输出端连续驱动。如果没有驱动元件连接到 wire 型变量，没有驱动器连接的时候，缺省值一般为 "Z"。。表示以assign语句赋值的组合逻辑信号。模块中的输入/输出信号类型缺省为wire型。
>
> 输出始终随输入的变化而变化的变量,例如硬连线。（组合逻辑）
>
> **wire**，tri：连线类型（两者功能一致） 
>
> wor，trior：具有线或特性的连线（两者功能一致） 
>
> wand，triand：具有线与特性的连线（两者功能一致） 
>
> tri1，tri0：上拉电阻和下拉电阻
>
> supply1，supply0：电源（逻辑1）和地（逻辑0）

```verilog
wire[n-1:0] a  //宽度为n的总线a; 
wire[n-1:0] 数据名1,数据名2, ……,数据名m; //m条宽度为n的总线
```

#### （2）reg

> 对应具有状态保持作用的电路元件（如触发器、寄存器等）,常用来表示过程块语句（如initial，always，task，function）内的指定信号 。它会保持数据原有的值，直到被改写。
>
> **register型变量与nets型变量的根本区别是： **
>
> **-  register型变量需要被明确地赋值，并且在被重新赋值前一直保持原值。** 
>
> **- register型变量必须通过过程赋值语句赋值！不能通过assign语句赋值！** 
>
> **- 在过程块内被赋值的每个信号必须定义成register型！**
>
> **- reg型变量既可生成触发器，也可生成组合逻辑；wire型变量只能生成组合逻辑。**

在过程块中被赋值的信号，往往代表触发器，但不一定就是触发器（也可以是组合逻辑信号）！ 

```verilog
reg[4:0] regc,regd; //regc,regd为5位宽的reg型向量
//组合逻辑
module rw1( a, b, out1, out2 ) ；
	input a, b；
	output out1, out2；
	reg out1；
	wire out2；
	assign out2 = a ；//连续赋值
	always @(b)
		out1 <= ~b；//过程赋值
endmodule 
//触发器
module rw2( clk, d, out1, out2 )；
	input clk, d；
	output out1, out2；
	reg out1；
	wire out2；
	assign out2 = d & ~out1 ;//连续赋值
  always @(posedge clk) //沿触发
		begin 
			out1 <= d ;//过程赋值
		end
endmodule
```

### 3. Memory

#### （1）向量

> 向量组织相近的信号，便于命名

> 当位宽大于 1 时，wire 或 reg 即可声明为向量的形式

```verilog
reg [3:0]      counter ;    //声明4bit位宽的寄存器counter
wire [32-1:0]  gpio_data;   //声明32bit位宽的线型变量gpio_data
wire [8:2]     addr ;       //声明7bit位宽的线型变量addr，位宽范围为8:2
reg [0:31]     data ;       //声明32bit位宽的寄存器变量data, 最高有效位为0
wire [9:0]     data_low = data[0:9] ;
addr_temp[3:2] = addr[8:7] + 1'b1 ;
reg [31:0]     data1 ;
reg [3:0]      byte1 [7:0];
integer j ;
always@* begin
  for (j=0; j<=7;j=j+1) begin
    byte1[j] = data1[(j+1)*3-1 : 3*8]; 
    //把data1[3:0]…data1[31:28]依次赋值给byte1[0][3:0]…byte[7][3:0]
    end
end
```

> **Verillog 还支持指定 bit 位后固定位宽的向量域选择访问。**
>
> - **[bit+: width]** : 从起始 bit 位开始递增，位宽为 width。
>
> - **[bit-: width]** : 从起始 bit 位开始递减，位宽为 width。

```verilog
//下面 2 种赋值是等效的
A = data1[31-: 8] ;
A = data1[31:24] ;
//下面 2 种赋值是等效的
B = data1[0+ : 8] ;
B = data1[0:7] ;
```

> **对信号重新进行组合成新的向量时，需要借助大括号。**

```verilog
wire [31:0]    temp1, temp2 ;
assign temp1 = {byte1[0][7:0], data1[31:8]};  //数据拼接
assign temp2 = {32{1'b0}};  //赋值32位的数值0  
```

#### （2）数组

> 一个n位的寄存器可用一条赋值语句赋值； 一个完整的存储器则不行！若要对某存储器中的存储单元进行读写操作，必须指明该单元在存储器中的地址！ 
>
> 在 Verilog 中允许声明 reg, wire, integer, time, real 及其向量类型的数组。数组维数没有限制。线网数组也可以用于连接实例模块的端口。数组中的每个元素都可以作为一个标量或者向量，以同样的方式来使用，形如：**<数组名>[<下标>]**。对于多维数组来讲，用户需要说明其每一维的索引。
>
> 向量是一个单独的元件，位宽为 n；数组由多个元件组成，其中每个元件的位宽为 n 或 1。它们在结构的定义上就有所区别。

```verilog
reg[n-1:0] rega；//一个n位的寄存器
reg[7:0] mema [3:0] ；//由4个8位寄存器组成的存储器
rega = 0； //合法赋值语句
mema = 0 ； //非法赋值语句
mema[8] = 1 ； //合法赋值语句
mema[1023:0] = 0 ；//合法赋值语句,对存储器大范围赋值

integer          flag [7:0] ; //8个整数组成的数组
reg  [3:0]       counter [3:0] ; //由4个4bit计数器组成的数组
wire [7:0]       addr_bus [3:0] ; //由4个8bit wire型变量组成的数组
wire             data_bit[7:0][5:0] ; //声明1bit wire型变量的二维数组
reg [31:0]       data_4d[11:0][3:0][3:0][255:0] ; //声明4维的32bit数据变量数组

flag [1]   = 32'd0 ; //将flag数组中第二个元素赋值为32bit的0值
counter[3] = 4'hF ;  //将数组counter中第4个元素的值赋值为4bit 十六进制数F，等效于counter[3][3:0] = 4'hF，即可省略宽度; 
assign addr_bus[0]        = 8'b0 ; //将数组addr_bus中第一个元素的值赋值为0
assign data_bit[0][1]     = 1'b1;  //将数组data_bit的第1行第2列的元素赋值为1，这里不能省略第二个访问标号，即 assign data_bit[0] = 1'b1; 是非法的。
data_4d[0][0][0][0][15:0] = 15'd3 ;  //将数组data_4d中标号为[0][0][0][0]的寄存器单元的15~0bit赋值为3
```

#### （3）存储器

存储器变量就是一种寄存器数组，可用来描述 RAM 或 ROM 的行为。

```verilog
reg               membit[0:255] ;  //256bit的1bit存储器
reg  [7:0]        mem[0:1023] ;    //1Kbyte存储器，位宽8bit
mem[511] = 8'b0 ;                  //令第512个8bit的存储单元值为0

parameter wordsize=16,memsize=256;         
reg [wordsize-1:0] mem[memsize-1:0],writereg, readreg;
```

### 4. 时间/整数/实数

- Verilog 使用特殊的时间寄存器 time 型变量，对仿真时间进行保存。其宽度一般为 64 bit，通过调用系统函数 $time 获取当前仿真时间。
- 整数类型用关键字 integer 来声明。声明时不用指明位宽，位宽和编译器有关，一般为32 bit。reg 型变量为无符号数，而 integer 型变量为有符号数。
- 实数用关键字 real 来声明，可用十进制或科学计数法来表示。实数声明不能带有范围，默认值为 0。如果将一个实数赋值给一个整数，则只有实数的整数部分会赋值给整数。

```verilog
time       current_time ;
initial begin
       #100 ;
       current_time = $time ; //current_time 的大小为 100
end
integer j ;  //整型变量，用来辅助生成数字电路
real    data1 ;
```

## 表达式

### 1. 运算符

#### （1）算数运算符

- **双目操作符**

对 2 个操作数进行算术运算：包括乘（*）、除（/）、加（+）、减（-）、求幂（**）、取模（%）

> 如果操作数某一位为 X，则计算结果也会全部出现 X。
>
> 对变量进行声明时，要根据变量的操作符对变量的位宽进行合理声明，不要让结果溢出。上述例子中，相加的 2 个变量位宽为 4bit，那么结果寄存器变量位宽最少为 5bit。否则，**高位将被截断**，导致结果高位丢失。无符号数乘法时，**结果变量位宽应该为 2 个操作数位宽之和**。

```verilog
a = 4'b0010 ;
b = 4'b100x ;
c = a+b ;       //结果为c=4'bxxxx
reg [5:0]        res ;
mula = 4'he   ;
mulb = 2'h3   ;
res  = mula * mulb ; //结果为res=6'h2a, 数据结果没有丢失位数
```

- 单目操作符

\+ 和 - 也可以作为单目操作符来使用，表示操作数的正负性，此类操作符优先级最高。

~ （按位取反）!（逻辑非）

> 负数表示时，可以直接在十进制数字前面增加一个减号 **-**，也可以指定位宽。因为负数使用二进制补码来表示，不指定位宽来表示负数，编译器在转换时，会自动分配位宽，从而导致意想不到的结果。例如

```verilog
mula = -4'd4 ;
mulb = 2 ;
res = mula * mulb ;      //计算结果为res=-6'd8, 即res=6'h38，正常
res = mula * (-'d4) ;    //(4的32次幂-4) * 2, 结果异常 
```

#### （2）逻辑运算符

> **进行逻辑运算后的结果为布尔值（为1或0或x）！**

&&(逻辑与)	||(逻辑或)	!(逻辑非)

#### （3）位运算符

> **两个不同长度的操作数进行位运算时，将自动按右端对齐，位数少的操作数会在高位用0补齐。** 
>
> A = 5’b11001，B = 3’b101， 
>
> 则A & B = （5’b11001）&（5’b**00**101）= 5’b00001

~ 按位取反	&按位与	|按位或	^按位异或	~^按位同或

#### （4）关系运算符

> **运算结果为1位的逻辑值1或0或x。关系运算时，若关系为真，则返回值为1；若声明的关系为假，则返回值为0；如果操作数中有一位为 x 或 z，则关系表达式的结果为 x。** 

<	<=	>	>= 

#### （5）等式运算符

> 使用等于运算符时，两个操作数必须逐位相等,结果才为1；若某些位为x或z，则结果为x。
>
> 使用全等运算符时，若两个操作数的相应位完全一致（如同是1，或同是0，或同是x，或同是z）,则结果为1；否则为0。

等价操作符包括逻辑相等（==），逻辑不等（!=），全等（===），非全等（!==）。

等价操作符的正常结果有 2 种：为真（1）或假（0）。

逻辑相等/不等操作符不能比较 x 或 z，当操作数包含一个 x 或 z，则结果为 x。

全等比较时，如果按位比较有相同的 x 或 z，返回结果也可以为 1，即全等比较可比较 x 或 z。所以，全等比较的结果一定不包含 x。

#### （6）缩减运算符

> **对单个操作数进行递推运算,即先将操作数的位0与位1进行与、或、非运算，再将运算结果与第三位进行相同的运算，依次类推，直至最高位 。**
>
> reg[3:0] a;
>
> b=|a; //等效于 b =( (a[0] | a[1]) | a(2)) | a[3]

归约操作符包括：归约与（&），归约与非（~&），归约或（|），归约或非（~|），归约异或（），归约同或（~^）。归约操作符只有一个操作数，它对这个向量操作数逐位进行操作，最终产生一个 1bit 结果。

#### （7）移位运算符

移位操作符包括左移（<<），右移（>>），算术左移（<<<），算术右移（>>>）。移位操作符是双目操作符，两个操作数分别表示要进行移位的向量信号（操作符左侧）与移动的位数（操作符右侧）。算术左移和逻辑左移时，右边低位会补 0。逻辑右移时，左边高位会补 0；而算术右移时，左边高位会补充符号位，以保证数据缩小后值的正确性。

#### （8）条件运算符

**信号 = 条件？表达式1:表达式2**

```verilog
module top_module(
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum
);
    
    wire[15:0] sum1,sum2;
    wire cout1;
    add16 a1(a[15:0] ,b[15:0] , 0, sum1,cout1);
    add16 a2(a[31:16] ,b[31:16] , cout1?1:0, sum2);
    assign sum = {sum2,sum1};

endmodule
```

#### （9）位拼接运算符

**{信号1的某几位，信号2的某几位， …… ，信号n的某几位}** 

```verilog
output [3:0] sum; //和
output cout; //进位输出
input[3:0] ina,inb;
input cin;
assign {cout,sum} = ina + inb +cin；//进位与和拼接在一起
  
{a,b[3:0],w,3’b101} = {a,b[3],b[2],b[1],b[0],w,1’b1,1’b0,1’b1}; //复杂的赋值
```

可用重复法简化表达式，如：{4{w}} 等同于{w,w,w,w}。还可用嵌套方式简化书写，如：{b,{3{a,b}}} 等同于{b,{a,b},{a,b},{a,b}}，也等同于{b,a,b,a,b,a,b}

## 模块

> 模块是 Verilog 中基本单元的定义形式，是与外界交互的接口。

```verilog
module module_name /*#(parameter_list)*/(port_list) ;
      Declarations_and_Statements ;
endmodule
```

![](https://www.runoob.com/wp-content/uploads/2020/09/jxRkciGWpiEbvz3D.png)

### 1. 端口

> 模块与外界交互的接口。对于外部环境来说，模块内部是不可见的，对模块的调用只能通过端口连接进行。

#### （1）**端口列表**

模块的定义中包含一个可选的端口列表，一般将不带类型、不带位宽的信号变量罗列在模块声明里。下面是一个 PAD 模型的端口列表：

```verilog
module pad(DIN, OEN, PULL,DOUT, PAD);
module test ;  //直接分号结束
```

#### （2）**端口声明**

根据端口的方向，端口类型有 3 种： 输入（input），输出（output）和双向端口（inout）。**input、inout 类型不能声明为 reg 数据类型，因为 reg 类型是用于保存数值的，而输入端口只能反映与其相连的外部信号的变化，不能保存这些信号的值。**output 可以声明为 wire 或 reg 数据类型。

在 Verilog 中，端口隐式的声明为 wire 型变量，即当端口具有 wire 属性时，不用再次声明端口类型为 wire 型。但是，当端口有 reg 属性时，则 reg 声明不可省略。

```verilog
//端口类型声明
input        DIN, OEN ;
input [1:0]  PULL ;     
inout        PAD ;     
output       DOUT ;    
reg          DOUT ;
//信号的声明完全可以合并成一句：
output reg      DOUT ;
//在 module 声明时就陈列出端口及其类型
module pad(
    input        DIN, OEN ,
    input [1:0]  PULL ,
    inout        PAD ,
    output reg   DOUT
    );
```

#### （3）端口连接规则

- **输入端口**：模块例化时，从模块外部来讲， input 端口可以连接 wire 或 reg 型变量。这与模块声明是不同的，从模块内部来讲，input 端口必须是 wire 型变量。

- **输出端口**：模块例化时，从模块外部来讲，output 端口必须连接 wire 型变量。这与模块声明是不同的，从模块内部来讲，output 端口可以是 wire 或 reg 型变量。

- **输入输出端口**：模块例化时，从模块外部来讲，inout 端口必须连接 wire 型变量。这与模块声明是相同的。

- **悬空端口**：模块例化时，如果某些信号不需要与外部信号进行连接交互，我们可以将其悬空，即端口例化处保留空白即可，上述例子中有提及。

> output 端口正常悬空时，我们甚至可以在例化时将其删除。
>
> input 端口正常悬空时，悬空信号的逻辑功能表现为高阻状态（逻辑值为 z）。但是，例化时一般不能将悬空的 input 端口删除，否则编译会报错，

#### （4）**位宽匹配**

当例化端口与连续信号位宽不匹配时，**端口会通过无符号数的右对齐或截断方式进行匹配。**

假如在模块 full_adder4 中，端口 a 和端口 b 的位宽都为 4bit，则下面代码的例化结果会导致：**u_adder4.a = {2'bzz, a[1:0]}, u_adder4.b = b[3:0]** 。

```verilog
full_adder4  u_adder4(
    .a      (a[1:0]),      //input a[3:0]
    .b      (b[5:0]),      //input b[3:0]
    .c      (1'b0),
    .so     (so),
    .co     (co));
```



### 2. 模块例化

> 在一个模块中引用另一个模块，对其端口进行相关连接，叫做模块例化。

#### （1）端口连接

- 命名端口连接

这种方法将需要例化的模块端口与外部信号按照其名字进行连接，端口顺序随意，可以与引用 module 的声明端口顺序不一致，只要保证端口名字与外部信号匹配即可。

如果某些输出端口并不需要在外部连接，例化时 可以悬空不连接，甚至删除。一般来说，input 端口在例化时不能删除，否则编译报错，output 端口在例化时可以删除。

```verilog
full_adder1  u_adder0(
    .Ai     (a[0]),
    .Bi     (b[0]),
    .Ci     (c==1'b1 ? 1'b0 : 1'b1),
    .So     (so_bit0),
    .Co     ())/*悬空或者不要/*);
```

- 顺序端口连接

这种方法将需要例化的模块端口按照模块声明时端口的顺序与外部信号进行匹配连接，位置要严格保持一致。例如例化一次 1bit 全加器的代码可以改为：

```verilog
full_adder1  u_adder1(a[1], b[1], co_temp[0], so_bit1, co_temp[1]);
```

#### （2）generate 模块例化

当例化多个相同的模块时，一个一个的手动例化会比较繁琐。用 generate 语句进行多个模块的重复例化，可大大简化程序的编写过程。

```verilog
module full_adder4(
    input [3:0]   a ,   //adder1
    input [3:0]   b ,   //adder2
    input         c ,   //input carry bit
 
    output [3:0]  so ,  //adding result
    output        co    //output carry bit
    );
 
    wire [3:0]    co_temp ; 
    //第一个例化模块一般格式有所差异，需要单独例化
    full_adder1  u_adder0(
        .Ai     (a[0]),
        .Bi     (b[0]),
        .Ci     (c==1'b1 ? 1'b1 : 1'b0),
        .So     (so[0]),
        .Co     (co_temp[0]));
 
    genvar        i ;
    generate
      for(i=1; i<=3; i=i+1) begin: adder_gen //记得for结构需要用begin开头，标识符adder_gen
        full_adder1  u_adder(
            .Ai     (a[i]),
            .Bi     (b[i]),
            .Ci     (co_temp[i-1]), //上一个全加器的溢位是下一个的进位
            .So     (so[i]),
            .Co     (co_temp[i]));
        end
    endgenerate
 
    assign co    = co_temp[3] ;
 
endmodule
```

```verilog
module top_module (
    input [31:0] a,
    input [31:0] b,
    output [31:0] sum
);//
    wire sum1[15:0],cout1[15:0];
    wire[15:0] sum2;
    
    add1  a0(
        .a     (a[0:0]),
        .b     (b[0:0]),
        .cin     (0), //上一个全加器的溢位是下一个的进位
        .sum     (sum1[0]),
        .cout     (cout1[0]));
    
    genvar        i ;
    generate
        for(i=1; i<=15; i=i+1) begin: add_gen
        add1  add_gen(
            .a     (a[i:i]),
            .b     (b[i:i]),
            .cin     (cout1[i-1]), //上一个全加器的溢位是下一个的进位
            .sum     (sum1[i]),
            .cout     (cout1[i]));
        end
    endgenerate
    
    add16 a2(a[31:16], b[31:16], cout1[15], sum2);    
    
    assign sum = {sum2,sum1[15],sum1[14],sum1[13],sum1[12],sum1[11],sum1[10],sum1[9],sum1[8],sum1[7],sum1[6],sum1[5],sum1[4],sum1[3],sum1[2],sum1[1],sum1[0]};
    //assign sum={sum2,{16{1'b0}}};
endmodule

module add1 ( input a, input b, input cin,   output sum, output cout );

// Full adder module here
    assign {cout,sum} = a+b+cin;

endmodule
```

#### （3）层次访问

每一个例化模块的名字，每个模块的信号变量等，都使用一个特定的标识符进行定义。在整个层次设计中，每个标识符都具有唯一的位置与名字。Verilog 中，通过使用一连串的 **.** 符号对各个模块的标识符进行层次分隔连接，就可以在任何地方通过指定完整的层次名对整个设计中的标识符进行访问。层次访问多见于仿真中。

```verilog
//u_n1模块中访问u_n3模块信号: 
a = top.u_m2.u_n3.c ;

//u_n1模块中访问top模块信号
if (top.p == 'b0) a = 1'b1 ; 

//top模块中访问u_n4模块信号
assign p = top.u_m2.u_n4.d ;
```

### 3. 带参数例化

当一个模块被另一个模块引用例化时，高层模块可以对低层模块的参数值进行改写。这样就允许在编译时将不同的参数传递给多个相同名字的模块，而不用单独为只有参数不同的多个模块再新建文件。

#### （1）defparam

可以用关键字 defparam 通过模块层次调用的方法，来改写低层次模块的参数值。例如对一个单口地址线和数据线都是 4bit 宽度的 ram 模块的 MASK 参数进行改写：

```verilog
module  ram_4x4
    (
     input               CLK ,
     input [4-1:0]       A ,
     input [4-1:0]       D ,
     input               EN ,
     input               WR ,    //1 for write and 0 for read
     output reg [4-1:0]  Q    );
    parameter        MASK = 3 ;
endmodule
defparam     u_ram_4x4.MASK = 7 ;
ram_4x4    u_ram_4x4
    (
        .CLK    (clk),
        .A      (a[4-1:0]),
        .D      (d),
        .EN     (en),
        .WR     (wr),    //1 for write and 0 for read
        .Q      (q)    );
```

#### （2）带参数模块例化

第二种方法就是例化模块时，将新的参数值写入模块例化语句，以此来改写原有 module 的参数值。

```verilog
ram #(.AW(4), .DW(4))
    u_ram
    (
        .CLK    (clk),
        .A      (a[AW-1:0]),
        .D      (d),
        .EN     (en),
        .WR     (wr),    //1 for write and 0 for read
        .Q      (q)
     );
```

> **建议，对已有模块进行例化并将其相关参数进行改写时，不要采用 defparam 的方法。除了上述缺点外，defparam 一般也不可综合。**
>
> **而且建议，模块在编写时，如果预知将被例化且有需要改写的参数，都将这些参数写入到模块端口声明之前的地方（用关键字井号 **#** 表示）。这样的代码格式不仅有很好的可读性，而且方便调试。**





## 语句

### 1. 赋值语句

#### （1）连续赋值语句

**assign语句，用于对wire型变量赋值**，是描述组合逻辑最常用的方法之一。 

```verilog 
assign LHS_target = RHS_expression  ；
assign c=a&b; //a、b、c均为wire型变量
```

- LHS_target 必须是一个标量或者线型向量，而不能是寄存器类型。
- RHS_expression 类型没有要求，可以是标量或线型或存器向量，也可以是函数调用。
- 只要 RHS_expression **表达式的操作数有事件发生（值的变化）时，RHS_expression 就会立刻重新计算，同时赋值给 LHS_target。**

Verilog 还提供了另一种对 wire 型赋值的简单方法，即在 wire 型变量声明的时候同时对其赋值。wire 型变量只能被赋值一次，因此该种连续赋值方式也只能有一次。例如下面赋值方式和上面的赋值例子的赋值方式，效果都是一致的。

```verilog
wire      A, B ;
wire      Cout = A & B ;
```

#### （2）过程赋值语句

用于对reg型变量赋值，有两种方式：

- 非阻塞（non-blocking)赋值方式：赋值符号为<=，如 b <= a ；
- 阻塞（blocking)赋值方式：赋值符号为=，如 b = a ；

```verilog
module ifblock(clk,i_a,o_b,o_c);
	input clk,i_a; 
	output o_b,o_c;
	reg b=0,c=0; 
	assign o_c=c;
	assign o_b=b; 
	always @(posedge clk) 
		begin 
			b<=i_a; //非阻塞赋值
			c<=b;
		end
endmodule
```

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/ruAMsa53pVQWN7FLK88i5pP4lFc9E8uwdoRm8sOFypQHbTHOF7GyULmAxT7r5r8KWvpPH2mcEihfDTwOyZCClA2pJC8w5OZVLuLWdWTY87Q!/b&bo=iAbCAQAAAAADB28!&rf=viewer_4)

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcZuC30kOzWKKihzb9B*jB1TVflblahaRCDcnvdWs*3vClif3scs*6ntDAy1C0yjSwUWGJ*a6S64PhTTmGAJUfc8!/b&bo=bgNmAwAAAAADFzo!&rf=viewer_4)

```verilog
module ifblock(clk,i_a,o_b,o_c);
	input clk,i_a; 
	output o_b,o_c;
	reg b=0,c=0; 
	assign o_c=c;
	assign o_b=b; 
	always @(posedge clk) 
		begin 
			b=i_a; //阻塞赋值
			c=b;
		end
endmodule
```

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcX9VaEoiEPq58M4xpZRLzddxHsE9ja9yYWCeUMN8Wzt2ve3CyHo7bMDMzkYfNKOIcuwV1RnSWEWHiCqqtVzUN6I!/b&bo=RAewAQAAAAADF8A!&rf=viewer_4)

![](http://m.qpic.cn/psc?/V10kiNUn1bcNEr/45NBuzDIW489QBoVep5mcX9VaEoiEPq58M4xpZRLzdcg*6OvPgNxVc9PqIHnvelgt2aZrecxPk0fv1biN3v2f0dR.oJZtk13h2baYLvl.7o!/b&bo=1ARoAwAAAAADF4k!&rf=viewer_4)

### 条件语句

#### （1）if-else语句

```verilog
if（表达式1） 语句1；
else if（表达式2）语句2；
  …
else if（表达式n）语句n；
  
if(sec_h==6) //第三层
	begin
		min_l=min_l+1;
		sec_h=0;
	end
```

#### （2）case语句

- 在case语句中，分支表达式每一位的值都是确定的（或者为0，或者为1）； 

- 在casez语句中，若分支表达式某些位的值为高阻值z，则不考虑对这些位的比较；

- 在casex语句中，若分支表达式某些位的值为z或不定值x，则不考虑对这些位的比较。

  在分支表达式中，可用？来标识x或z

```verilog
always@ (select[3:0] or a or b or c or d)
	begin
		casez (select)
			4’b???1: out = a； 
      4’b??1? : out = b;
      4’b? 1?? : out = c;
      4’b 1??? : out = d;
    endcase
	end
```

#### （3）for语句

```verilog
for （表达式1；表达式2；表达式3）语句
  
integer i;
always @(vote)
	begin
		sum = 0; //sum初值为0
		for(i = 0;i<=6;i = i+1) //for语句
			if(vote[i]) sum = sum+1;
			//只要有人投赞成票，则 sum加1
			if(sum[2]) pass = 1; //若超过4人赞成，则表决通过
			else pass = 0;
	end
```

#### （4）repeat语句

```verilog
repeat （循环次数表达式）
	begin
	……
	end
```

#### （5）while语句

```verilog
while （循环执行条件表达式）
  begin
  ……
  end
```

### 3. 结构说明语句

> 过程结构语句有 2 种，initial 与 always 语句。它们是行为级建模的 2 种基本语句。一个模块中可以包含多个 initial 和 always 语句，但 2 种语句不能嵌套使用。这些语句在模块间并行执行，与其在模块的前后顺序没有关系。但是 initial 语句或 always 语句内部可以理解为是顺序执行的（非阻塞赋值除外）。每个 initial 语句或 always 语句都会产生一个独立的控制流，执行时间都是从 0 时刻开始。

#### （1）**always语句**

always 语句是重复执行的。always 语句块从 0 时刻开始执行其中的行为语句；当执行完最后一条语句后，便再次执行语句块中的第一条语句，**如此循环反复**。

```verilog
module test ;
 
    parameter CLK_FREQ   = 100 ; //100MHz
    parameter CLK_CYCLE  = 1e9 / (CLK_FREQ * 1e6) ;   //switch to ns
 
    reg  clk ;
    initial      clk = 1'b0 ;      //clk is initialized to "0"
    always     # (CLK_CYCLE/2) clk = ~clk ;       //generating a real clock by reversing
 
    always begin
        #10;
        if ($time >= 1000) begin
            $finish ;
        end
    end
 
endmodule
```



```verilog
always @ (<敏感信号表达式>)
  begin
  // 过程赋值语句
  // if语句
  // case语句
  // while，repeat，for循环
  // task，function调用
  end
```

- **敏感信号表达式又称事件表达式或敏感表，当其值改变时，则执行一遍块内语句；典型的敏感信号是时钟！**

- **敏感信号可以为单个信号，也可为多个信号，中间需用关键字or连接！** **敏感信号不要为x或z！**

- always的时间控制可以为沿触发，也可为电平触发。 关键字posedge表示上升沿；negedge表示下降沿。

- **当always块有多个敏感信号时，一定要采用if - elseif语句，而不能采用并列的if语句！否则易造成一个寄存器有多个时钟驱动，将出现不能综合的错误。**

```verilog
always @ (posedge min_clk or negedge reset)
  begin
  	if (reset)
  		min<=0;
  	else if (min==8’h59) //当reset无效且min=8’h59时
  		begin
  			min<=0;
  			h_clk<=1;
  		end
	end
```

#### （2）initial语句

**在仿真的初始状态对各变量进行初始化；** **在测试文件中生成激励波形作为电路的仿真信号。**

- initial 语句从 0 时刻开始执行，只执行一次，多个 initial 块之间是相互独立的。

- 如果 initial 块内包含多个语句，需要使用关键字 begin 和 end 组成一个块语句。如果 initial 块内只要一条语句，关键字 begin 和 end 可使用也可不使用。

- initial 理论上来讲是不可综合的，多用于初始化、信号检测等。

```verilog
initial begin
	inputs = ’b000000; 
  #10 inputs = ’b011001; 
  #10 inputs = ’b011011; 
  #10 inputs = ’b011000; 
end

//at proper time stop the simulation
initial begin
  forever begin
    #100;
    //$display("---gyc---%d", $time);
    if ($time >= 1000) begin
      $finish ;
    end
  end
end
```

#### （3）task和function语句

task和function语句分别用来由用户定义任务和函数。任务和函数往往是大的程序模块中在不同地点多次用到的相 同的程序段。

- task

当希望能够对一些信号进行一些运算并输出多个结果（即有多个输出变量）时，宜采用任务结构。

```verilog
task my_task;
	input a,b;
	inout c;
	output d,e;
  ……
  <语句> //执行任务工作相应的语句
  ……
  c = foo1; d = foo2; //对任务的输出变量赋值
  e = foo3;
endtask

my_task（v,w,x,y,z);
//当任务启动时，由v 、w和x传入的变量赋给了a、b和c； 
//当任务完成后，输出通过c、d和e赋给了x、y和z
```

- 函数

函数的目的是通过返回一个用于某表达式的值，来响应输入信号。适于对不同变量采取同一运算的操作。

函数在模块内部定义，通常在本模块中调用，也能根据按模块层次分级命名的函数名从其他模块调用。而任务只能在同一模块内定义与调用！

```verilog
function[7:0] gefun; //函数的定义
  input [7:0] x;
  <语句> //进行运算
  gefun = count; //赋值语句
endfunction
assign number = gefun(rega); //对函数的调用
```

> 函数的定义不能包含任何时间控制语句——用延迟#、事件控制@或等待wait标识的语句。 
>
> 函数不能启动（即调用）任务！ 
>
> 定义函数时至少要有一个输入参量！且不能有任何输出或输入/输出双向变量。 
>
> 在函数的定义中必须有一条赋值语句，给函数中的一个内部寄存器赋以函数的结果值，该内部寄存器与函数同名。

### 4. 编译预处理语句

#### （1）‵define语句

用一个指定的标志符（即宏名）来代表一个字符串（即宏内容）。 

```verilog
module test;
  reg a,b,c;
  wire out; 
  ‵define aa a + b 
  ‵define cc c +‵
  aa //引用已定义的宏名‵
  aa 来定义宏cc
  assign out = ‵cc;
```

#### （2）**`undef** 语句

用来取消之前的宏定义

```verilog
`define    DATA_DW     32

`undef DATA_DW
```

#### （3）‵include语句

文件包含语句——一个源文件可将另一个源文件的全部内容包含进来。 

```verilog
`include         "../../param.v"
`include         "header.v"
```

#### （4）‵timescale语句

时间尺度语句——用于定义跟在该命令后模块的时间单位和时间精度。

```
`timescale      time_unit / time_precision
```

time_unit 表示时间单位，time_precision 表示时间精度，它们均是由数字以及单位 s（秒），ms（毫秒），us（微妙），ns（纳秒），ps（皮秒）和 fs（飞秒）组成。时间精度可以和时间单位一样，但是时间精度大小不能超过时间单位大小，例如下面例子中，输出端 Z 会延迟 5.21ns 输出 A&B 的结果。

```verilog
`timescale 1ns/100ps    //时间单位为1ns，精度为100ps，合法
//`timescale 100ps/1ns  //不合法
module AndFunc(Z, A, B);
    output Z;
    input A, B ;
    assign #5.207 Z = A & B
endmodule
```

#### （5）条件编译语句

```
`ifdef, `ifndef, `elsif, `else, `endif

```
- 如果定义了 MCU51，则使用第一种参数说明；如果没有定义 MCU、定义了 WINDOW，则使用第二种参数说明；如果 2 个都没有定义，则使用第三种参数说明。

```verilog
`ifdef       MCU51
    parameter DATA_DW = 8   ;
`elsif       WINDOW
    parameter DATA_DW = 64  ;
`else
    parameter DATA_DW = 32  ;
`endif
```

- 也可用 **`ifndef** 来设置条件编译，表示如果没有相关的宏定义，则执行相关语句。下面例子中，如果定义了 WINDOW，则使用第二种参数说明。如果没有定义 WINDOW，则使用第一种参数说明。 

```verilog
`ifndef     WINDOW
    parameter DATA_DW = 32 ;  
 `else
    parameter DATA_DW = 64 ;
 `endif
```

#### （6）`default_nettype语句

该指令用于为隐式的线网变量指定为线网类型，即将没有被声明的连线定义为线网类型。

```verilog
`default_nettype wand 
`default_nettype none
//Z1 无定义就使用，系统默认Z1为wire型变量，有 Warning 无 Error
module test_and(
        input      A,
        input      B,
        output     Z);
    assign Z1 = A & B ;  
endmodule
```

#### （7）`resetall语句

该编译器指令将所有的编译指令重新设置为缺省值。**`resetall`** 可以使得缺省连线类型为线网类型。当 **`resetall`** 加到模块最后时，可以将当前的 **`timescale`** 取消防止进一步传递，只保证当前的 **`timescale`** 在局部有效，避免 ``timescale` 的错误继承。

#### （8）`celldefine语句

这两个程序指令用于将模块标记为单元模块，他们包含模块的定义。例如一些与、或、非门，一些 PLL 单元，PAD 模型，以及一些 Analog IP 等。

```verilog
`celldefine
module (
    input      clk,
    input      rst,
    output     clk_pll,
    output     flag);
        ……
endmodule
`endcelldefine
```

#### （9）`unconnected_drive语句

在模块实例化中，出现在这两个编译指令间的任何未连接的输入端口，为正偏电路状态或者为反偏电路状态。

```verilog
`unconnected_drive pull1
. . .
 / *在这两个程序指令间的所有未连接的输入端口为正偏电路状态（连接到高电平） * /
`nounconnected_drive
`unconnected_drive pull0
. . .
 / *在这两个程序指令间的所有未连接的输入端口为反偏电路状态（连接到低电平） * /
`nounconnected_drive 
```

### 5. 时延语句

#### （1）时延

连续赋值延时语句中的延时，**用于控制任意操作数发生变化到语句左端赋予新值之间的时间延时。**

**时延一般是不可综合的。**寄存器的时延也是可以控制的，这部分在时序控制里加以说明。连续赋值时延一般可分为普通赋值时延、隐式时延、声明时延。

```verilog
//普通时延，A&B计算结果延时10个时间单位赋值给Z
wire Z, A, B ;
assign #10    Z = A & B ;
//隐式时延，声明一个wire型变量时对其进行包含一定时延的连续赋值。
wire A, B;
wire #10        Z = A & B;
//声明时延，声明一个wire型变量是指定一个时延。因此对该变量所有的连续赋值都会被推迟到指定的时间。除非门级建模中，一般不推荐使用此类方法建模。
wire A, B;
wire #10 Z ;
assign           Z =A & B
```

#### （2）惯性时延

在上述例子中，A 或 B 任意一个变量发生变化，那么在 Z 得到新的值之前，会有 10 个时间单位的时延。如果在这 10 个时间单位内，即在 Z 获取新的值之前，A 或 B 任意一个值又发生了变化，**那么计算 Z 的新值时会取 A 或 B 当前的新值。所以称之为惯性时延，即信号脉冲宽度小于时延时，对输出没有影响。**

因此仿真时，时延一定要合理设置，防止某些信号不能进行有效的延迟。

```verilog
module time_delay_module(
    input   ai, bi,
    output  so_lose, so_get, so_normal);
 
    assign #20      so_lose      = ai & bi ;
    assign  #5      so_get       = ai & bi ;
    assign          so_normal    = ai & bi ;
endmodule
```

testbench

```verilog
`timescale 1ns/1ns
module test ;
    reg  ai, bi ;
    wire so_lose, so_get, so_normal ;

    initial begin
        ai        = 0 ;
        #25 ;      ai        = 1 ;
        #35 ;      ai        = 0 ;        //60ns
        #40 ;      ai        = 1 ;        //100ns
        #10 ;      ai        = 0 ;        //110ns
    end
 
    initial begin
        bi        = 1 ;
        #70 ;      bi        = 0 ;
        #20 ;      bi        = 1 ;
    end
 
    time_delay_module  u_wire_delay(
        .ai              (ai),
        .bi              (bi),
        .so_lose         (so_lose),
        .so_get          (so_get),
        .so_normal       (so_normal));
 
    initial begin
        forever begin
            #100;
            //$display("---gyc---%d", $time);
            if ($time >= 1000) begin
                $finish ;
            end
        end
    end
 
endmodule
```

- 信号 so_normal 为正常的与逻辑。

- 由于所有的时延均大于 5ns，所以信号 so_get 的结果为与操作后再延迟 5ns 的结果。

- 信号 so_lose 前一段是与操作后再延迟 20ns 的结果。由于信号 ai 第二个高电平持续时间小于 20ns，so_lose 信号会因惯性时延而漏掉对这个脉冲的延时检测，所以后半段 so_lose 信号仍然为 0。

![](https://www.runoob.com/wp-content/uploads/2020/09/3NVRY0qgsLe27LZS.png)