SRCS1 := hello.c
OBJS1 := $(filter %.o,${SRCS1:.c=.o})
SRCS2 := hello2.c utils.c
OBJS2 := $(filter %.o,${SRCS2:.c=.o})
SRCS3 := hello3.c
OBJS3 := $(filter %.o,${SRCS3:.c=.o})
SRCS4 := hello4.c
OBJS4 := $(filter %.o,${SRCS4:.c=.o})
SRCSL := utils.c utils2.c
OBJSL := $(filter %.o,${SRCSL:.c=.o})

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

all: hello hello2 hello3 hello4
hello: $(OBJS1)
	$(CC) $(CFLAGS) -o hello $(OBJS1)

hello2: $(OBJS2)
	$(CC) $(CFLAGS) -o hello2 $(OBJS2)

libutils.a: $(OBJSL)
	$(AR) r libutils.a $(OBJSL)

hello3: libutils.a $(OBJS3)
	$(CC) $(CFLAGS) -o hello3 $(OBJS3) -L. -lutils

hello4: libutils.a $(OBJS4)
	$(CC) $(CFLAGS) -o hello4 $(OBJS4) -L. -lutils

clean:
	rm -f $(OBJS1) $(OBJS2) $(OBJSL) $(OBSJ3) hello hello2 hello3 hello4 libutils.a
