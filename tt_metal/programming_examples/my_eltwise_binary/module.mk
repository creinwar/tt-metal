ELTWISE_BINARY_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/my_eltwise_binary/my_eltwise_binary.cpp

ELTWISE_BINARY_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/my_eltwise_binary.d

-include $(ELTWISE_BINARY_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/my_eltwise_binary
$(PROGRAMMING_EXAMPLES_TESTDIR)/my_eltwise_binary: $(PROGRAMMING_EXAMPLES_OBJDIR)/my_eltwise_binary.o $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/my_eltwise_binary.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/my_eltwise_binary.o: $(ELTWISE_BINARY_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
