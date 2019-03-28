-module(tarry).
-export([start/0]).

%%
%% MAIN PROCESS
%%

start() ->
    % Read the initiator and topology data from stdin
    {Initiator, Topology} = read_input(),
    io:format("Initiator: ~p~n", [Initiator]),
    io:format("Topology: ~p~n", [Topology]),

    % Spawn the nodes required
    Nodes = spawn_nodes(Topology),

    % Inform the node processes about the others that they are connected to
    % Uses acks to ensure that all connections are set up before the algorithm begins
    send_connections(Topology, Nodes),

    % Get the Pid of the Initiator
    {_, InitiatorPid} = lists:keyfind(Initiator, 1, Nodes),

    % Perform Tarry algorithm
    Token = do_tarry(InitiatorPid),

    % Print the solution to stdout
    io:format("Tarry Solution:~n~s~n", [string:join(Token, " ")]).

read_input() ->
    % Read the initiator as the only thing in the first line
    [Initiator|_] = read_input_line(),
    % Read the network topology from the remaining lines
    Topology = read_topology([]),
    {Initiator, Topology}.

read_input_line() ->
    case io:get_line("") of
        eof ->
            % End of the file
            eof;
        Data ->
            % Trim the trailing newline and split the line about the spaces
            string:tokens(string:strip(Data, right, $\n), " ")
    end.

read_topology(Topology) ->
    case read_input_line() of
        eof ->
            % End of the file, return the network Topology accumulated
            Topology;
        [Node|ConnectedNodes] ->
            % Add the line to the Topology accumulator and recurse
            read_topology([{Node, ConnectedNodes}|Topology])
    end.

spawn_nodes(Topology) ->
    % Spawn a node for each row in the Topology, returning a list of {NodeName, NodePid} pairs
    [spawn_node(NodeName) || {NodeName, _} <- Topology].

spawn_node(NodeName) ->
    % Spawn one node from the main function of the tarry_node module
    Pid = spawn(tarry_node, main, [self(), NodeName]),
    % Return a pair of the node's name given in the input file and the corresponding process ID
    {NodeName, Pid}.

send_connections([], _) ->
    % Base case, all connections have been sent
    ok;

send_connections([{NodeName, ConnectedNodeNames}|TopologyTl], Nodes) ->
    % Find the Pid of the current node NodeName from the Nodes list of pairs
    {_, NodePid} = lists:keyfind(NodeName, 1, Nodes),
    % Find the list of nodes that the current node NodeName is connected to
    ConnectedNodes = [lists:keyfind(N, 1, Nodes) || N <- ConnectedNodeNames],
    % Send this list of connected nodes to the current node's process at NodePid
    NodePid ! {connections, ConnectedNodes},
    % Wait for an acknowledgement that the process has received the list before recursing
    % This makes sure that all processes receive their connected nodes before we continue to the Tarry algorithm
    receive
        {ack} ->
            send_connections(TopologyTl, Nodes)
    end.

do_tarry(InitiatorPid) ->
    % Send the first tarry message to the initiator to start the algorithm
    InitiatorPid ! {tarry, self(), []},
    % Wait to receive the final tarry message back from the initiator with the final answer in the Token
    receive
        {tarry, _, Token} ->
            % Reverse the Token because it was accumulated with cons
            lists:reverse(Token)
    end.
