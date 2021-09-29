## 文件编码

### 1. byte与Byte

- byte是java的**基本数据类型，存储整型数据**，占据1个字节(8 bits)，能够存储的数据范围是-128～+127

- Byte是java.lang中的一个类，目的是为**基本数据类型byte进行封装**。

- Byte是byte的**包装类**,就如同Integer和int的关系，一般情况包装类用于**泛型或提供静态方法，用于基本类型或字符串之间转换**，建议尽量不要用包装类和基本类型之间运算,因为这样运算效率会很差的。

- 包装类是对象，拥有方法和字段，对象的调用都是通过引用对象的地址；基本类型不是。包装类型是引用的传递；基本类型是值的传递 

![](https://ss2.baidu.com/6ON1bjeh1BF3odCf/it/u=2973106743,2881514465&fm=15&gp=0.jpg)

### 2. 文件的编码

> gbk编码中文占用两个字节，英文一个字节
>
> utf-8编码中文占用三个字节，英文一个字节
>
> java是双字节编码，utf-16be（中英都是双字节)

```java
String str = "超人不会飞ABC";
//转换为字节用的项目默认编码utf-8
byte[] bytes = str.getBytes();
for(byte b:bytes){
  /*
   *把字节以16进制显示
   *因为Integer.toHexString()的接收参数是int,不是byte,于是运算是会先把byte强制转换为int
   *由于java中强制转换是保持值不变,而在计算机中数都是用补码表示的,java中int是32位4个byte
   *5 00000000 00000000 00000000 00000101 正 反 补
   *-5 10000000 00000000 00000000 00000101 正
   *-5 11111111 11111111 11111111 11111010 反（反码为对该数的原码除符号位外各位取反。）
   *-5 11111111 11111111 11111111 11111011 补 （反码加1）0xFFFFFFFB(四位一个）
   *& 0xff 排除前面1的干扰
   */
  System.out.print(Integer.toHexString(b & 0xff)+" ");
  //output:	e8 b6 85 e4 ba ba e4 b8 8d e4 bc 9a e9 a3 9e 41 42 43 
}

byte[] bytes1 = str.getBytes("gbk");
for(byte b:bytes1){
  System.out.print(Integer.toHexString(b & 0xff)+" ");
  //output:b3 ac c8 cb b2 bb bb e1 b7 c9 41 42 43 
}

System.out.println();
byte[] bytes2 = str.getBytes("utf-16be");
for(byte b:bytes2){
  System.out.print(Integer.toHexString(b & 0xff)+" ");
  //output:8d 85 4e ba 4e d 4f 1a 98 de 0 41 0 42 0 43 
}

String str2 = new String(bytes2);
System.out.println(str2); //output:O��ABC
String str3 = new String(bytes2,"utf-16be");
System.out.println(str3);//output:超人不会飞ABC
```

## 磁盘操作

### 1. File

> Java.io.file用于表示文件/目录，只表示信息（大小，名称），不用于文件访问
>
> api详解文档

### 2. RandomAccessFile

> java.io.RandomAccessFile提供对文件内容访问，既可以读，又可以写
>
> 支持随机访问文件，支持访问文件任意位置（文件指针，打开文件时，pointer=0在开头）
>
> 两种模式 "rw"读写，"r"只读

- 写方法：`raf.write(int)` 只写一个字节，指针移动下一个位置(或者字节数组)。
- 读方法：`raf.read() `读一个字节(或者字节数组)。
- 文件读写完成后，必须关闭，否则会有错误`raf.close()`。

```java
File file = new File("1.txt");
if(!file.exists()||file.isFile()){
  file.createNewFile();
}
RandomAccessFile randomAccessFile = new RandomAccessFile(file,"rw");
System.out.println(randomAccessFile.getFilePointer()); //output:0
randomAccessFile.write('A');//只写一个字节
System.out.println(randomAccessFile.getFilePointer()); //output:1
//写四次，每次只写后面8位
int i = 0x7fffffff;
randomAccessFile.write(i >>> 24); //高8位
randomAccessFile.write(i >>> 16);
randomAccessFile.write(i >>> 8);
randomAccessFile.write(i );
System.out.println(randomAccessFile.getFilePointer()); //output:5
//直接写int,底层就是上面
randomAccessFile.writeInt(i);
String str = "中";
byte[] gbk = str.getBytes("gbk");
randomAccessFile.write(gbk);
//读文件，指针开头
randomAccessFile.seek(0);
byte[] buff = new byte[(int)randomAccessFile.length()];
randomAccessFile.read(buff);
System.out.println(Arrays.toString(buff));
//output:[65, 127, -1, -1, -1, 127, -1, -1, -1, -42, -48]
randomAccessFile.close();
```

## 字节流

> InputStream、OutputStram抽象类。EOF=END，读到-1结束。流用完要关闭in.close()。
>
> Java I/O 使用了装饰者模式来实现。FilterInputStream 属于抽象装饰者，装饰组件为组件提供额外的功能。

![](https://camo.githubusercontent.com/d650ccc4ec1a0c99171582d9ccc9a5003155496f/68747470733a2f2f63732d6e6f7465732d313235363130393739362e636f732e61702d6775616e677a686f752e6d7971636c6f75642e636f6d2f39373039363934622d646230352d346363652d386432662d3163386230396634643932312e706e67)

### 1. InputStream

- `int b = in.read()`读取一个字节到int低八位（-1是EOF）。
- `in.read(byte[] buf) `读取数据填充到字节数组。
- `in.read(byte[] buf,int start,int size)` 从start位置开始存放size长度数据。

### 2. OutputStram

- `out.write(int b)` 将b的低八位写入流。
- `out.write(byte[] buf)` 写入字节数组到流。
- `out.write(byte[] buf,int start,int size)` 从start位置开始写入size长度数据到流。

### 3. File(In/Out)putStream

- `new FileOutputStream("name")` **如果不存在则创建，如果存在则删除后再创建（覆盖原内容）**。
- `new FileOutputStream("name",true)` **如果不存在则创建，如果存在则在原文件后面追加内容（不覆盖）**。

```java
//copy文件操作，批量读写

InputStream fileInputStream = new FileInputStream("1.txt");
OutputStream fileOutputStream = new FileOutputStream("2.txt");
byte[] buffer = new byte[20 * 1024];//20k大小,作为复制流的缓冲区大小
int cnt;
/*
 *read() 最多读取 buffer.length 个字节，返回的cnt是实际读取的个数 
 * 1读不满：就把0-cnt的字节写入，下一次读就是-1
 * 2不够放：先放length长度，下一次继续读取
 *返回 -1 的时候表示读到 eof，即文件尾
 */
while ((cnt = fileInputStream.read(buffer, 0, buffer.length)) != -1) {
  fileOutputStream.write(buffer, 0, cnt);
}
fileInputStream.close();
fileOutputStream.close();
```

### 4. Data(In/Out)putStream

- 对流的扩展，更方便写int,long,string等，包装了这些的字节操作。

```java
OutputStream out = new DataOutputStream(new FileOutputStream("2.txt"));
((DataOutputStream) out).writeInt(10);
((DataOutputStream) out).writeUTF("这个");
((DataOutputStream) out).writeChars("china");
out.close();
InputStream in = new DataInputStream(new FileInputStream("2.txt"));
System.out.println(((DataInputStream) in).readInt());//output:10
System.out.println(((DataInputStream) in).readUTF());//output:这个
System.out.println(((DataInputStream) in).readUTF());//exception读取失败
in.close();
```

### 5. Buffered(In/Out)putStream

- 提供缓存区，提高了IO性能，**减少了硬盘IO次数**（类比于桶中倒水到另一个桶）。
  - File：一滴一滴把水转移过去。
  - Data：一瓢一瓢把水转移。
  - Buffered：先把水一瓢一瓢放入缓冲区，再倒入另一个桶。

- File模式，是程序从硬盘上读取一个字节字后，再写入一个字节，然后再读取再写入，**因为磁盘io的速度是非常慢的，所以耗时较长**。缓冲方法内部根据一定的算法**在内存中开辟一个空间-缓冲区**，读取一个（或者若干个）字节之后，先放入内存缓冲区，然后写入的时候，**从缓冲区中写入硬盘**。因为内存的io速度非常快，因此可以更为高效的利用硬盘。
- **批量缓冲读取效率更高，批量读写可能要调用几次buf，而批量缓冲读取是最厉害的，把字节一部分一部分的拿出，每次拿出都放入字节数组，再将字节数组放入缓冲区中，再一次性将缓冲区中的内容写到磁盘文件上。**

```java
//拷贝：批量缓冲读取

bufferedInputStream = new BufferedInputStream(new FileInputStream(srcFile));
bufferedOutputStream = new BufferedOutputStream(new FileOutputStream(destFile));
byte[] bytes = new byte[8 * 1024]; //字节数组
int length = 0;
while ((length = bufferedInputStream.read(bytes, 0, bytes.length)) != -1) {
  bufferedOutputStream.write(bytes, 0, length);
  outputStream.flush(); //一定要刷新
}
//关闭流
```

## 字符流

> 文本(char)是16位无符号整数，是字符unicode编码（双字节）。unicode编码为每一个「字符」分配一个唯一的 ID（学名为码位 / 码点 / Code Point）
>
> 文件是byte byte byte字节序列。
>
> 文本文件是是文本(char)按某种编码方案(utf8,gbk)序列化的byte byte byte字节序列。编码规则：将「码位」转换为字节序列的规则（编码/解码 可以理解为 加密/解密 的过程）。

### 1.InputStreamReader/OutputStreamWriter

- 以前是按字节写进去，现在是按字符写入。

```java
InputStreamReader reader = new InputStreamReader(new FileInputStream("1.txt"),"utf8");
//默认项目编码
OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream("2.txt"));
int c;
char[] buf = new char[24];
while((c=reader.read(buf,0,buf.length))!=-1){
  writer.write(buf,0,c);
}
writer.close();
reader.close();
```

### 2. FileWriter/FileReader

```java
FileReader reader = new FileReader("1.txt");
FileWriter writer = new FileWriter("2.txt");//true则添加不覆盖
int c;
char[] buf = new char[24];
while((c=reader.read(buf,0,buf.length))!=-1){
  writer.write(buf,0,c);
}
reader.close();
writer.close();
```

### 3. BufferedReader/Bufferedwriter

- 提供缓冲功能。`BufferedReader`用于加快读取字符的速度，`BufferedWriter`用于加快写入的速度。
- 当`BufferedReader`在读取文本文件时，会先尽量从文件中读入字符数据并放满缓冲区，而之后若使用read()方法，会先从缓冲区中进行读取。如果缓冲区数据不足，才会再从文件中读取。
- 使用`BufferedWrite`r时，写入的数据并不会先输出到目的地，而是先存储至缓冲区中。如果缓冲区中的数据满了，才会一次对目的地进行写出。
- `FileReader`能一次读取一个字符，或者一个字符数组。而`BufferedReader`也可以，同时`BufferedReader`还能一次读取一行字符串。同时,`BufferedReader`带缓冲，会比`FileReader`快很多。

```java
BufferedReader reader = new BufferedReader(new FileReader("1.txt"));
BufferedWriter writer = new BufferedWriter(new FileWriter("2.txt"));
String line;
while((line=reader.readLine())!=null){
  writer.write(line);//不会识别换行符，写进去就是一行
  //单独换行
  writer.newLine();
}
reader.close();
writer.close();
```

### 4. PrintWriter

- PrintWriter相对于BufferedWriter的好处在于，如果PrintWriter开启了自动刷新，那么当PrintWriter调用println，prinlf或format方法时，输出流中的数据就会自动刷新出去。PrintWriter不但能接收字符流，也能接收字节流。
- Socket编程中,尽量用PrintWriter取代BufferedWriter，下面是PrintWriter的优点：
  - PrintWriter的print、println方法可以接受任意类型的参数，而BufferedWriter的write方法只能接受字符、字符数组和字符串；
  - PrintWriter的println方法自动添加换行，BufferedWriter需要显示调用newLine方法；
  - PrintWriter的方法不会抛异常，若关心异常，需要调用checkError方法看是否有异常发生；
  - PrintWriter构造方法可指定参数，实现自动刷新缓存（autoflush）；PrintWriter的构造方法更广。

```java
BufferedReader reader = new BufferedReader(new FileReader("1.txt"));
PrintWriter writer = new PrintWriter("2.txt");
String line;
while((line=reader.readLine())!=null){
  writer.println(line); //换行操作方便
}
reader.close();
writer.close();
```

## 对象序列化

> 将object转化为byte序列为序列化。
>
> 只有实现了 Serializable 或 Externalizable 接口的类的对象才能被序列化，否则抛出异常！
>
> 不会对静态变量进行序列化，因为序列化只是保存对象的状态，静态变量属于类的状态。
>
> 当一个对象的实例变量引用其他对象，序列化该对象时也把引用对象进行序列化。
>
> 序列化时，只对对象的状态进行保存，而不管对象的方法。
>
> 常见序列化协议：XML、JSON、SOAP等。

### 1. Object(Out/In)putStream

```java
//序列化 writeObject
Student s = new Student("alex",15);
ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream("student.dat"));
objectOutputStream.writeObject(s);
objectOutputStream.close();
//反序列化 readObject
ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream("student.dat"));
Student s2 = (Student) objectInputStream.readObject();
objectInputStream.close();
```

### 2. transient

- static和transient类型的成员数据不能被序列化。static代表类的状态，transient代表对象的临时数据。

- ArrayList 中存储数据的数组 elementData 是用 transient 修饰的，因为这个数组是动态扩展的，并不是所有的空间都被使用，因此就不需要所有的内容都被序列化。**通过重写序列化和反序列化方法，使得可以只序列化数组中有内容的那部分数据。**

```java
private transient Object[] elementData;

s.defalutWriteObject(); //把默认的属性序列化
for(int i=0;i<size;i++){ //自定义有效的元素序列化
  s.writeObject(elementData[i]);
}
```

